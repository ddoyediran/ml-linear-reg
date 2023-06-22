import dotenv from "dotenv";
dotenv.config();
import * as tf from "@tensorflow/tfjs";

// Load data from the csv file where consume is the label
const loadData = async () => {
  try {
    //   const dataUrl = "fuelcar.csv";

    const dataUrl = process.env.DATA_URL;

    const trainingData = tf.data.csv(dataUrl, {
      columnConfigs: {
        consume: {
          isLabel: true,
        },
      },
    });

    // Load the number of features (in this case only the first column will be used as the feature)
    const numberOfFeatures = (await trainingData.columnNames()).length - 1;
    //const numFeatures = (await csvDataset.columnNames()).length - 1;

    //console.log("Working here");

    // prepare the Dataset for training.
    // const flattenedDataset = trainingData.map(({ xs, ys }) => {
    //   // Convert xs(features) and ys(labels) from object form (keyed by
    //   // column name) to array form.
    //   return {
    //     distanceData: Object.values(xs),
    //     consume: Object.values(ys),
    //   };
    // });

    // console.log(await trainingData.toArray());
    //console.log(await flattenedDataset.toArray());
    //console.log(numberOfFeatures);

    const mapData = await trainingData.toArray();

    let trainingDataset = {
      distance: [],
      consume: [],
    };

    for (let i = 0; i < mapData.length; i++) {
      trainingDataset.distance.push(mapData[i].xs.distance);
      trainingDataset.consume.push(mapData[i].ys.consume);
    }

    console.log(trainingDataset);
  } catch (e) {
    console.error(e.message);
  }
};

loadData();
