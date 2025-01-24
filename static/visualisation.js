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

// The code of this visualization is a modified version of the circle packing tutorial
// published here: https://observablehq.com/@d3/pack/2

// Specify the dimensions of the chart.
const height = 800;
const width = height;
const margin = 20; // to avoid clipping the root circle stroke

// Specify the number format for values.
const format = d3.format(",d");

// Create the pack layout.
const pack = d3.pack()
    .size([width - margin * 2, height - margin * 2])
    .padding(4);

// Compute the hierarchy from the JSON data; recursively sum the
// values for each node; sort the tree by descending value; lastly
// apply the pack layout.
const root = pack(d3.hierarchy(data)
    .sum(d => d.value)
    .sort((a, b) => b.value - a.value));

// Create the SVG container.
const svg = d3.create("svg")
    .attr("width", width)
    .attr("height", height)
    .attr("viewBox", [-margin, -margin, width, height])
    .attr("style", "max-height: 100%; max-width: 100%; font: 10px sans-serif;")
    .attr("text-anchor", "middle");

// Place each node according to the layout’s x and y values.
const node = svg.append("g")
    .selectAll()
    .data(root.descendants())
    .join("g")
    .attr("transform", d => `translate(${d.x},${d.y})`);

// Add a title.
node.append("title")
    .text(d => `${d.ancestors().map(d => d.data.name).reverse().join("/")}\n${format(d.value)}`);

const colorPalette = [
        "#ee6640",
        "#f88e53",
        "#fcb368",
        "#fed382",
        "#feeb9f",
        "#f8f7ae",
        "#e6f49d",
        "#cae983",
        "#a9da70",
        "#84ca66",
    ].reverse();

const colorScale = d3.scaleQuantize()
    .domain([0, Math.max(...data.children.map(child => child.value))])    // Radius domain
    .range(colorPalette);

// Add a filled or stroked circle.
node.append("circle")
    .attr("fill", d =>  d.children == null ?
        createRadialGradient(colorScale(d.r)) : "#fefef0") // Use the gradient
    .attr("stroke", d => d.children == null ?
        d3.color(colorScale(d.r)).darker(0.2) : "#fff")
    .attr("r", d => d.r);

node.filter(d => !d.children) // consider leaf nodes and non-leaf nodes as well now
    .append("foreignObject")
    .attr("x", d => -0.8 * d.r) // center horizontally
    .attr("y", d => -1*d.r) // center vertically, manually adjusted
    .attr("width", d => 1.6 * d.r)
    .attr("height", d => 1.8 * d.r)
    .append("xhtml:div")
    .classed("foreignDiv", true)
    .style("font-size", d => d.r / 4.5 + "px") // Dynamic font sizing
    .html(d =>
        "<span style='font-size: " + (d.r / 2.4) + "px; color: " + d3.color(colorScale(d.r)).darker(1) + ";'>"
        + format(d.value)
        + "</span>"
        + d.data.name
        + "<br/>"
    );

d3.select("#chart").node().appendChild(svg.node());

// Generate a radial gradient
function createRadialGradient(color) {
        const gradient = svg.append("defs")
            .append("radialGradient")
            .attr("id", (d) => "gradient-" + color.slice(1)) // Unique ID based on color
            .attr("cx", "50%") // Center of the gradient
            .attr("cy", "50%")
            .attr("r", "50%"); // Radius of the gradient

        gradient.append("stop")
            .attr("offset", "0%") // Start of the gradient (center)
            .attr("stop-color", d3.color(color).brighter(0.5)); // Slightly lighter color at the center

        gradient.append("stop")
            .attr("offset", "100%") // End of the gradient (edge)
            .attr("stop-color", color); // Original color at the edge

        return "url(#gradient-" + color.slice(1) + ")";
}