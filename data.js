// import * as dfd from "danfojs-node";
import * as tf from "@tensorflow/tfjs";

// Load data from the csv file where consume is the label
const loadData = async () => {
  //   const dataUrl = "fuelcar.csv";
  const blob = new Blob(["fuelcar.csv"], { type: "application/csv" });
  const dataUrl = URL.createObjectURL(blob);

  console.log(dataUrl);
  const trainingData = tf.data.csv(dataUrl, {
    columnConfigs: {
      consume: {
        isLabel: true,
      },
    },
  });

  // Load the number of features (in this case only the first column will be used as the feature)
  const numberOfFeatures = (await trainingData.columnNames()).length - 1;

  console.log(numberOfFeatures);
};

loadData();
