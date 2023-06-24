// import dotenv from "dotenv";
// dotenv.config();

// import * as tf from "@tensorflow/tfjs";
// import * as tfvis from "@tensorflow/tfjs-vis";

import { splitData } from "./data.js";

// require("dotenv").config();
// const tf = require("@tensorflow/tfjs");
// const tfvis = require("@tensorflow/tfjs-vis");
// const { splitData } = require("./data.js");

// Import the dataset

// Develop the linear regression model
const createModel = () => {
  const model = tf.sequential();

  // add the input layer
  model.add(tf.layers.dense({ inputShape: [1], units: 1, unitBias: true }));

  // add the output layer (optional)
  model.add(tf.layers.dense({ units: 1 }));

  return model;
};

// Run the model and Convert the dataset to tensor data type format
const trainAndRunModel = async () => {
  try {
    //console.log(await splitData());
    const data = await splitData();

    // const data = {
    //   trainDistance: [
    //     28, 12, 11.2, 12.9, 18.5, 8.3, 7.8, 12.3, 4.9, 11.9, 12.4, 11.8, 12.3,
    //     24.7, 12.4, 17.3, 33.4, 11.8, 25.9, 11.8, 25.3, 14.2, 17.9, 11.8, 12.3,
    //     12.4, 18.4, 18.4, 18.3, 18.4, 12.3, 11.8, 12.3, 32.6, 19, 12.1,
    //   ],

    //   trainConsume: [
    //     5, 4.2, 5.5, 3.9, 4.5, 6.4, 4.4, 5, 6.4, 5.3, 5.6, 4.6, 5.9, 5.1, 4.7,
    //     5.1, 5.6, 5.1, 4.9, 4.7, 5.5, 5.9, 5.7, 4.7, 5.9, 4.1, 5.7, 5.8, 5.5,
    //     5.7, 5.3, 5, 5.6, 4.8, 4.3, 5.7,
    //   ],

    //   testDistance: [
    //     11.8, 12.3, 2, 13.9, 9.7, 11.6, 14.2, 11.8, 24.8, 12.4, 34.8, 14.2, 5.2,
    //     10.5, 12.3, 11.8, 12.3, 13.2, 13, 12.9, 13.9, 11.8, 12.2, 12.5, 12.4,
    //     11.8, 11.8, 12.5, 15.7, 12.9,
    //   ],

    //   testConsume: [
    //     4.5, 5.2, 6.2, 5.1, 3.9, 5, 5.4, 4.5, 5.1, 4.7, 4, 5.4, 4.5, 3.6, 5.2,
    //     4.9, 6.2, 4.3, 5, 5.1, 5.6, 4.3, 5.8, 4, 4.7, 5.9, 5.3, 4.2, 5.3, 5.7,
    //   ],
    // };

    // convert dataset to tensor
    const tensorTrainDistance = tf.tensor1d(data.trainDistance);
    const tensorTestDistance = tf.tensor1d(data.testDistance);
    const tensorTrainTarget = tf.tensor1d(data.trainConsume);
    const tensorTestTarget = tf.tensor1d(data.testConsume);

    const model = createModel(); // create an instance of the model

    //   tfvis.show.modelSummary({ name: "Model summary" }, model);

    model.compile({ optimizer: "sgd", loss: "meanAbsoluteError" });

    model.summary();

    tfvis.show.modelSummary({ name: "Model Architecture" }, model);

    const batchSize = 35;
    const epochs = 40;

    const fitCallbacks = tfvis.show.fitCallbacks(
      { name: "Training Performance" },
      ["loss", "mae"],
      { height: 200, callbacks: ["onEpochEnd"] }
    );

    await model.fit(tensorTrainDistance, tensorTrainTarget, {
      batchSize,
      epochs,
      shuffle: true,
      // callbacks: tfvis.show.fitCallbacks(
      //   { name: "Training Performance" },
      //   ["loss", "mae"],
      //   { height: 200, callbacks: ["onEpochEnd"] }
      // ),
      // callbacks: {
      //   onEpochEnd: async (epoch, logs) => {
      //     console.log("Epoch: " + epoch, " Loss: " + logs.loss);
      //   },
      // },
      callbacks: [
        fitCallbacks,
        {
          onEpochEnd: async (epoch, logs) => {
            console.log("Epoch: " + epoch, " Loss: " + logs.loss);
          },
        },
      ],
    });

    // Evaluate the model
    const evaResult = await model.evaluate(
      tensorTestDistance,
      tensorTestTarget,
      {
        batchSize,
      }
    );

    evaResult.print();

    // Perform model prediction
    const result = await model.predict(tf.tensor1d([60]));

    result.print();

    //return;
  } catch (e) {
    console.error(e.message);
  }
};

trainAndRunModel();
