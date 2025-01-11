/*
 * Copyright 2025 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package smartfire;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.data.message.SystemMessage;
import dev.langchain4j.data.message.UserMessage;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.chat.ChatLanguageModel;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.output.Response;
import dev.langchain4j.model.vertexai.VertexAiEmbeddingModel;
import dev.langchain4j.model.vertexai.VertexAiGeminiChatModel;
import org.apache.commons.math3.ml.clustering.Clusterable;
import org.apache.commons.math3.ml.clustering.DBSCANClusterer;

import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.WebSocket;
import java.time.Instant;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CompletionStage;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class FirehoseConsumer {

    public static final int MAX_BATCH_SIZE = 250;
    public static final int EMBEDDING_OUTPUT_DIMENSION = 128;

    public static final int MINIMUM_POINTS_PER_CLUSTER = 10;
    public static final double MAXIMUM_NEIGHBORHOOD_RADIUS = 0.2;

    public static final int NUMBER_OF_MESSAGES = 10_000;
    public static final String LANG = "en";

    public static final String EMBEDDING_MODEL = "text-embedding-005";
//    public static final String EMBEDDING_MODEL = "text-multilingual-embedding-002";
    public static final String CHAT_MODEL = "gemini-1.5-flash-002";
//    public static final String CHAT_MODEL = "gemini-2.0-flash-exp";

    private static final EmbeddingModel embeddingModel = VertexAiEmbeddingModel.builder()
        .project(System.getenv("GCP_PROJECT_ID"))
        .location(System.getenv("GCP_LOCATION"))
        .endpoint(System.getenv("GCP_VERTEXAI_ENDPOINT"))
        .modelName(EMBEDDING_MODEL)
        .outputDimensionality(EMBEDDING_OUTPUT_DIMENSION)
        .publisher("google")
        .build();

    private static final ChatLanguageModel chatModel = VertexAiGeminiChatModel.builder()
        .project(System.getenv("GCP_PROJECT_ID"))
        .location(System.getenv("GCP_LOCATION"))
        .modelName(CHAT_MODEL)
        .maxOutputTokens(25)
        .build();

    public record Message(Commit commit, String did) {
        record Commit(Record record, String cid) {
            record Record(
                String text,
                List<String> langs,
                Date createdAt) {
            }
        }
    }

    private static final Gson GSON = new GsonBuilder()
        .setDateFormat("yyyy-MM-dd'T'HH:mm:ss.SSS'Z'")
        .create();

    private record ClusterableEmbeddedMessage(Message message, double[] embeddingVector) implements Clusterable {
        @Override
        public double[] getPoint() {
            return embeddingVector;
        }
    }

    private List<Message> liveMessages(int numberOfMessages, String lang) {
        int maxNumberOfMessages = numberOfMessages > 0 ? numberOfMessages : MINIMUM_POINTS_PER_CLUSTER;
        String language = lang == null ? LANG : lang;

        List<Message> messages = new ArrayList<>();
        AtomicInteger counter = new AtomicInteger();

        try (var httpClient = HttpClient.newHttpClient()) {
            httpClient.newWebSocketBuilder()
                .buildAsync(
                    URI.create("wss://jetstream2.us-east.bsky.network/subscribe?wantedCollections=app.bsky.feed.post"),
                    new WebSocket.Listener() {
                        @Override
                        public CompletionStage<?> onText(WebSocket webSocket, CharSequence data, boolean last) {
                            webSocket.request(1);
                            return CompletableFuture.completedFuture(data)
                                .thenAccept(text -> {
                                    if (counter.get() % 1000 == 0) System.out.format("Received %d messages%n", counter.get());
                                    Message message = GSON.fromJson(String.valueOf(text), Message.class);
                                    if (message.commit().record().langs().contains(language)) {
                                        if (!message.commit().record().text().isBlank()) {
                                            messages.add(message);
                                            if (counter.incrementAndGet() >= maxNumberOfMessages) {
                                                webSocket.abort();
                                            }
                                        }
                                    }
                                });
                        }
                    })
                .join();
        }
        return messages;
    }

    public static void main(String[] args) throws FileNotFoundException {
        StringBuilder dataTemplate = new StringBuilder();

        Instant start = Instant.now();

        // ----------------------------------------------
        // Collections messages from the Bluesky Firehose

        List<Message> allMessages = new FirehoseConsumer()
            .liveMessages(NUMBER_OF_MESSAGES, "en");

        Instant consummedInstant = Instant.now();
        System.out.format("Consumed %d messages in: %dms%n", allMessages.size(),
            consummedInstant.toEpochMilli() - start.toEpochMilli());

        // ------------------
        // Embed all messages

        List<TextSegment> allSegments = allMessages.stream()
            .map(message -> TextSegment.from(message.commit().record().text()))
            .toList();

        // calculate embedding batch sizes
        // (Vertex AI embeddings are batched by 250)
        int segmentsSize = allSegments.size();
        int numberOfParallelBatches = segmentsSize / MAX_BATCH_SIZE;

        // embed in parallel
        List<Embedding> allEmbeddings = IntStream.range(0, numberOfParallelBatches)
            .parallel()
            .mapToObj(i ->
                embeddingModel.embedAll(allSegments.subList(i * MAX_BATCH_SIZE, (i + 1) * MAX_BATCH_SIZE)).content()
            )
            .flatMap(List::stream)
            .toList();

        Instant embeddedInstance = Instant.now();
        System.out.format("Embedded messages in: %dms%n", embeddedInstance.toEpochMilli() - consummedInstant.toEpochMilli());

        // --------------------------------------------
        // Cluster all the messages in different groups
        // (using Apache Commons Maths clustering library)

        var clusterableEmbeddedMessages = IntStream.range(0, allEmbeddings.size())
            .mapToObj(i -> new ClusterableEmbeddedMessage(
                allMessages.get(i),
                allEmbeddings.get(i)
                    .vectorAsList()
                    .stream()
                    .mapToDouble(Float::doubleValue)
                    .toArray()))
            .toList();

        var clusters = new DBSCANClusterer<ClusterableEmbeddedMessage>(MAXIMUM_NEIGHBORHOOD_RADIUS, MINIMUM_POINTS_PER_CLUSTER)
            .cluster(clusterableEmbeddedMessages);

//        var clusters = new KMeansPlusPlusClusterer<ClusterableEmbeddedMessage>(
//            10, 10_000, new EuclideanDistance(), new JDKRandomGenerator(),
//            KMeansPlusPlusClusterer.EmptyClusterStrategy.FARTHEST_POINT)
//            .cluster(clusterableEmbeddedMessages);

        Instant clusteredInstant = Instant.now();
        System.out.format("Clustered messages in %d clusters in: %dms%n",
            clusters.size(),
            clusteredInstant.toEpochMilli() - embeddedInstance.toEpochMilli());

        for (int i = 0; i < clusters.size(); i++) {
            var cluster = clusters.get(i);
            var points = cluster.getPoints();

            System.out.format("%n=== Cluster #%d === (%d points) ===%n", i, points.size());

            String appendedMessages = points.stream()
                .map(clusteredMsg ->
                    clusteredMsg.message.commit().record().text()
                )
                .collect(Collectors.joining("\n"));

            Instant startSummarizationInstant = Instant.now();

            // Ask Gemini to come up with a summary for all the messages in a cluster
            // (those messages are supposed to be about the same topic)

            Response<AiMessage> modelResponse = chatModel.generate(
                SystemMessage.from("""
                    Summarize the following list of social media messages in one simple description.
                    Don't give a full sentence saying the social messages are about a topic,
                    just give the topic directly in 10 words or less,
                    without mentioning the messages are social media posts or reactions.
                    """),
                UserMessage.from(appendedMessages)
            );

            System.out.println(modelResponse.content().text());

            Instant generatedSummaryInstant = Instant.now();
            System.out.format("Summaries generated in: %dms%n", generatedSummaryInstant.toEpochMilli() - startSummarizationInstant.toEpochMilli());

            // In case the summary is too long, use an ellipsis
            String clusterSummary = switch (modelResponse.finishReason()) {
                case LENGTH -> modelResponse.content().text().trim() + "...";
                default -> modelResponse.content().text();
            };

            dataTemplate
                .append("{name: \"")
                .append(clusterSummary
                    .replace("\"", "\\\"")
                    .replace("\n", " "))
                .append("\", value: ")
                .append(points.size())
                .append("},\n    ");

            System.out.println("--- Messages: ");
            System.out.println(points.stream()
                .map(clusteredMsg ->
                    clusteredMsg.message.commit().cid() + " --> " +
                        clusteredMsg.message.commit().record().text()
                )
                .collect(Collectors.joining("\n")));
        }

        String output = String.format("""
            const data = {
              name: "Bluesky topic clusters",
              children: [
                %s
              ]
            };
            """, dataTemplate.toString().trim());

        System.out.println(output);

        // Write the JavaScript data to a file
        // which will be loaded by D3.js for visualisation
        PrintWriter writer = new PrintWriter("static/newdata.js");
        writer.append(output);
        writer.close();
    }
}