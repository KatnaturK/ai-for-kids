
let mobilenet_model;
let coco_ssd_model;
let verticalBoxArr = []
let inputImg;
let srcImg;
let count = 70;

let layer_index_list = [
    1, 3,
    10, 12,
    16, 18,
    22, 24,
    28, 30,
    34, 36,
    40, 42,
    46, 48,
    52, 54,
    58, 60,
    64, 66,
    70, 72
    // 76, 78
]

let layer_name_list = [
    "conv_1", "relu_1",
    "conv_2", "relu_2",
    "conv_3", "relu_3",
    "conv_4", "relu_4",
    "conv_5", "relu_5",
    "conv_6", "relu_6",
    "conv_7", "relu_7",
    "conv_8", "relu_8",
    "conv_9", "relu_9",
    "conv_10", "relu_10",
    "conv_11", "relu_11",
    "conv_12", "relu_12"
    // "conv_13", "relu_13"
]

window.addEventListener('load', function () {
    loadCocoSsdModel();
    loadMobilnetModel();

    $('#cnn-popup-close').click(function () {
        $('#cnn-popup').hide();
    });

    $("#model-summary").click(function () {
        $('#model-summary').hide();
    })
});

window.addEventListener('resize', function () {
    // loadModel().then(data => this.model = data);
});

function modelSummaryClick() {
    $("#model-summary").show();
}

async function loadCocoSsdModel() {
    coco_ssd_model = await cocoSsd.load();
}

async function loadMobilnetModel() {
    const model = await tf.loadLayersModel('https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json');
    this.model = model;
    loadVerticalBoxes();
}

async function loadVerticalBoxes() {
    var cnnDiv = document.getElementById('cnn-flow');
    cnnDiv.innerHTML = '';

    var verticalBox = document.createElement('div');
    verticalBox.className = 'vertical-box';
    cnnDiv.appendChild(verticalBox);
    loadBoxes(verticalBox, "in", 1);

    for (var i = 0; i <= count; i++) {
        if (layer_index_list.includes(i)) {
            var verticalBox = document.createElement('div');
            verticalBox.className = 'vertical-box';
            verticalBox.id = 'vertical-box-' + i;
            cnnDiv.appendChild(verticalBox);

            var verticalBoxLabel = document.createElement('p');
            verticalBoxLabel.className = 'vertical-box-label';
            verticalBoxLabel.id = 'vertical-box-label-' + i;
            verticalBox.appendChild(verticalBoxLabel);

            let x = this.model.layers[i].outputShape[this.model.layers[i].outputShape.length - 1];
            loadBoxes(verticalBox, i, x);
        }
    }

    var verticalBox = document.createElement('div');
    verticalBox.className = 'vertical-box';
    cnnDiv.appendChild(verticalBox);
    loadBoxes(verticalBox, "out", 1);
}

async function loadBoxes(verticalBox, id, boxCount) {
    let boxArr = []
    for (var i = 0; i < boxCount; i++) {
        var box = document.createElement('div');
        box.className = 'box';
        box.id = 'box-' + id + '-' + i;
        verticalBox.appendChild(box);
        let imgTensor = document.createElement('img');
        imgTensor.id = 'img-tensor-' + id + '-' + i;
        imgTensor.className = 'img-tensor-' + boxCount;
        box.appendChild(imgTensor);

        if (id != 'in' && id != 'out') {
            box.addEventListener('click', boxClick);
        }

        boxArr.push(box.id);
        Plotly.purge(box.id);
    }
    verticalBoxArr[id] = boxArr;
}

Array.prototype.reshape = function (rows, cols) {
    var copy = this.slice(0);
    this.length = 0;
    for (var r = 0; r < rows; r++) {
        var row = [];
        for (var c = 0; c < cols; c++) {
            var i = r * cols + c;
            if (i < copy.length) {
                row.push(copy[i]);
            }
        }
        this.push(row);
    }
}

function drawconv_map(x, elements_name, reshape_x, reshape_y, width_plot, height_plot) {

    var boxPlotDiv = document.getElementById(elements_name);

    x.reshape(reshape_x, reshape_y);
    var data = [
        {
            z: x,
            type: 'heatmap',
            colorscale: 'RdBu',
            // colorscale: [
            //     ['0.0', 'rgb(255,0,0)'],
            //     ['0.5', 'rgb(0,255,0)'],
            //     ['1.0', 'rgb(0,05,255)'],
            // ],
            showlegend: false,
            showarrow: false,
            showscale: false,
            showgrid: false
        }
    ];

    var layout = {
        autosize: false,
        width: width_plot,
        height: height_plot,
        margin: {
            l: 0,
            r: 0,
            b: 0,
            t: 0,
            pad: 0
        },
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        showlegend: false,
        xaxis: { visible: false },
        yaxis: { visible: false },

    };

    Plotly.newPlot(elements_name, data, layout, { staticPlot: true });
}

async function predictUsingCocoSsd() {
    const inImg = document.getElementById("prediction-in-img");
    var predictions = await coco_ssd_model.detect(inImg);

    createPredictionTable(predictions);
    drawBoundingBoxes(predictions);

    document.getElementById("info").style.visibility = "visible";
}

function predictUsingMobilenet() {
    const inImg = tf.browser.fromPixels(document.getElementById("img-tensor-in-0"));
    const feedImage = tf.image.resizeBilinear(inImg, [224, 224]);

    let input = []
    input.push(tf.tidy(() => { return tf.expandDims(feedImage, 0).asType('float32').div(255.0) }));

    for (let i = 1; i <= count; i++) {
        input.push(this.model.layers[i].apply(input[i - 1]));
    }

    for (let i = 1; i <= count; i++) {
        if (layer_index_list.includes(i)) {

            document.getElementById("vertical-box-label-" + i).style.visibility = "visible";
            document.getElementById("vertical-box-label-" + i).innerText = layer_name_list[layer_index_list.indexOf(i)];

            const conv = input[i];
            const conv_list = tf.tidy(() => { return tf.unstack(conv.reshape([conv.shape[1], conv.shape[2], conv.shape[3]]), 2) });

            for (let j = 0; j < conv.shape[3]; j++) {
                const reverse_img = tf.reverse2d(conv_list[j]);
                verticalBoxArr[i][j] = {
                    reverse_img: reverse_img,
                    shape_i: conv.shape[1],
                    shape_j: conv.shape[2],
                    shape_k: conv.shape[3]
                };
                drawconv_map(Array.from(reverse_img.dataSync()), "box-" + i + "-" + j, verticalBoxArr[i][j].shape_i, verticalBoxArr[i][j].shape_j, 40, 40);
                // reverse_img.dispose();

                document.getElementById("box-" + i + "-" + j).style.visibility = "visible";
            }
        }
    }
}

function uploadImageChange(event) {
    inputImg = event;
    if (inputImg !== undefined && inputImg.target.files.length > 0) {
        $("#loader-popup").show();

        loadMobilnetModel();

        document.getElementById("prediction").style.visibility = "hidden";
        document.getElementById("cnn").style.visibility = "hidden";
        document.getElementById("cnn-flow").style.visibility = "hidden";
        document.getElementById("prediction-in").style.visibility = "hidden";
        document.getElementById("prediction-out").style.visibility = "hidden";
        document.getElementById("prediction-stats").style.visibility = "hidden";
        document.getElementById("box-" + "in-" + 0).style.visibility = "hidden";
        document.getElementById("box-" + "out-" + 0).style.visibility = "hidden";
        document.getElementById("info").style.visibility = "hidden";

        document.getElementById("prediction-in-img").src = "";
        document.getElementById("prediction-out-img").src = "";
        document.getElementById("img-tensor-in-0").src = "";
        document.getElementById("img-tensor-out-0").src = "";

        srcImg = URL.createObjectURL(inputImg.target.files[0]);
        document.getElementById("prediction").style.visibility = "visible";
        document.getElementById("prediction-in").style.visibility = "visible";
        document.getElementById("prediction-in-img").src = srcImg;


        $("#loader-popup").hide();
    }
}

function uploadImageClick() {
    document.getElementById("input-img").click();
}

function predictImageClick() {
    if (document.getElementById("prediction-in-img").src != "") {

        predictUsingCocoSsd();

        document.getElementById("prediction-out").style.visibility = "visible";
        document.getElementById("prediction-stats").style.visibility = "visible";
        document.getElementById("cnn").style.visibility = "visible";
    }
}

function cnnFlowClick() {
    $("#loader-popup").show();
    var cnnFlowDiv = document.getElementById("cnn-flow");
    var cnnFlowStyle = getComputedStyle(cnnFlowDiv);

    if (cnnFlowStyle.visibility == "hidden") {
        document.getElementById("cnn-flow").style.visibility = "visible";
        document.getElementById("box-" + "in-" + 0).style.visibility = "visible";
        document.getElementById("box-" + "out-" + 0).style.visibility = "visible";

        document.getElementById("img-tensor-in-0").src = srcImg;
        document.getElementById("img-tensor-out-0").src = srcImg;

        predictUsingMobilenet();
    }
    $("#loader-popup").hide();
}

function boxClick(event) {
    $("#cnn-popup").show();

    for (let i = 1; i <= 6; i++) {
        for (let j = 1; j <= 3; j++) {
            Plotly.purge('cnn-popup-box-' + i + '-' + j);
        }
    }

    var boxDiv = this;
    x = parseInt(boxDiv.id.split("-")[1]);
    y = parseInt(boxDiv.id.split("-")[2]);
    if (x % 3 == 0)
        x -= 2;
    x = layer_index_list.indexOf(x) % 8;
    y -= 2;
    if (y <= 0)
        y = 1;

    var tmp_x = [];
    for (let i = 0; i < 3; i++) {
        tmp_x[i * 2 + 1] = layer_index_list[x];
        tmp_x[i * 2 + 2] = layer_index_list[x + 1];
        x += 8;
    }

    for (let i = 1; i <= 6; i++) {
        let k = y % verticalBoxArr[tmp_x[i]][0].shape_k;

        for (let j = 1; j <= 3; j++) {
            const reverse_img = verticalBoxArr[tmp_x[i]][k].reverse_img;
            drawconv_map(
                Array.from(reverse_img.dataSync()),
                "cnn-popup-box-" + i + "-" + j,
                verticalBoxArr[tmp_x[i]][k].shape_i,
                verticalBoxArr[tmp_x[i]][k].shape_j,
                100, 100
            );
            k++;
        }
        document.getElementById('cnn-img-txt-' + i).innerHTML = layer_name_list[layer_index_list.indexOf(tmp_x[i])];
    }
}

function createPredictionTable(predictions) {
    var predictionTable = new Array();
    predictionTable.push(["Prediction Id", "Prediction Label", "Confidence Score"]);

    for (let i = 0; i < predictions.length; i++) {
        var score = Math.round(predictions[i].score * 100);
        predictionTable.push([i + 1, predictions[i].class, score + "%"]);
    }

    var table = document.createElement("TABLE");
    table.border = "1";

    var row = table.insertRow(-1);
    for (var i = 0; i < predictionTable[0].length; i++) {
        var headerCell = document.createElement("TH");
        headerCell.innerHTML = predictionTable[0][i];
        row.appendChild(headerCell);
    }

    for (var i = 1; i < predictionTable.length; i++) {
        row = table.insertRow(-1);
        for (var j = 0; j < predictionTable[0].length; j++) {
            var cell = row.insertCell(-1);
            cell.innerHTML = predictionTable[i][j];
        }
    }

    var dvTable = document.getElementById("prediction-stats-table");
    dvTable.innerHTML = "";
    dvTable.appendChild(table);
}

function drawBoundingBoxes(predictions) {
    var inImg = document.getElementById("prediction-in-img");
    var outImg = document.getElementById("prediction-out-img")
    var imgWidth = outImg.width;
    var imgHeight = outImg.height;
    outImg.src = inImg.src;

    var outImgRect = outImg.getBoundingClientRect()

    var canvas = document.getElementById("prediction-out-canvas");
    canvas.width = outImgRect.width;
    canvas.height = outImgRect.height;
    canvas.style.top = outImgRect.top;
    canvas.style.left = outImgRect.left;

    var ctx = canvas.getContext("2d");

    const font = "16px sans-serif";
    ctx.font = font;
    ctx.textBaseline = "top";
    ctx.width = imgWidth;
    ctx.height = imgHeight;

    predictions.forEach((prediction, index) => {

        const x = prediction.bbox[0];
        const y = prediction.bbox[1];
        const width = prediction.bbox[2];
        const height = prediction.bbox[3];
        ctx.strokeStyle = "#00FFFF";
        ctx.lineWidth = 2;
        ctx.strokeRect(x, y, width, height);

        ctx.fillStyle = "#00FFFF";
        const textWidth = ctx.measureText(prediction.class).width;
        const textHeight = parseInt(font, 10);
        ctx.fillRect(x, y, textWidth + 12, textHeight + 4);
    });

    predictions.forEach((prediction, index) => {
        const x = prediction.bbox[0];
        const y = prediction.bbox[1];
        ctx.fillStyle = "#000000";
        let label = (index + 1) + "." + prediction.class;
        ctx.fillText(label, x, y);
    });
}
