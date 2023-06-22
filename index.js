import dotenv from "dotenv";
dotenv.config();

import * as tf from "@tensorflow/tfjs";
import { splitData } from "./data.js";

// Import the dataset

// Convert the dataset to tensor data type format
async function trainModel() {
  //console.log(await splitData());
  const data = await splitData();

  // convert dataset to tensor
  const tensorTrainDistance = tf.tensor1d(data.trainDistance);
  const tensorTestDistance = tf.tensor1d(data.testDistance);
  const tensorTrainTarget = tf.tensor1d(data.trainConsume);
  const tensorTestTarget = tf.tensor1d(data.testConsume);

  tensorTestDistance.print();
}

trainModel();

// Develop the linear regression model
