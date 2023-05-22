// neural networks
import * as tf from '@tensorflow/tfjs';
import * as tfvis from '@tensorflow/tfjs-vis';
import * as dfd from 'danfojs';
import * as sk from 'scikitjs';
import { DecisionTreeRegressor } from 'scikitjs';
sk.setBackend(tf);

export const createModel = async (
  batchsize,
  epochs,
  optimizer = 'adam',
  learningRate = 0.1,
  inputs,
  output,
  valinputs,
  valoutput,
  testinputs,
  testoutput,
  type,
  elementA,
  elementB
) => {
  console.log(optimizer, learningRate);
  sk.setBackend(tf);
  // Xtrain , ytrain
  let Xtrain = new dfd.DataFrame(inputs);
  let ytrain = new dfd.DataFrame(output);
  Xtrain = Xtrain.tensor;
  ytrain = ytrain.tensor;
  // XVal , yVal
  let XVal = new dfd.DataFrame(valinputs);
  let yVal = new dfd.DataFrame(valoutput);
  XVal = XVal.tensor;
  yVal = yVal.tensor;
  // XTest , YTest
  console.log('test');
  console.log(inputs);
  console.log(output);
  console.log(valinputs);
  console.log(valoutput);
  console.log(testinputs);
  console.log(testoutput);
  let XTest = new dfd.DataFrame(testinputs);
  let yTest = new dfd.DataFrame(testoutput);
  XTest = XTest.tensor;
  yTest = yTest.tensor;
  //
  // Xtrain = await Xtrain.arraySync();
  // ytrain = await ytrain.dataSync();
  // XVal = await XVal.arraySync();
  // yVal = await yVal.dataSync();
  // console.log(Xtrain);
  // console.log(ytrain);
  // DecisionTree(Xtrain, ytrain, XVal, yVal, XTest, yTest);
  // Create a sequential model
  const model = tf.sequential();
  console.log(type);
  // Add a single input layer

  model.add(
    tf.layers.dense({
      inputShape: Xtrain.shape[1],
      units: 64,
      useBias: true,
      activation: 'relu',
      kernelInitializer: 'heNormal',
    })
  );
  let trainLogs;
  let result;
  if (type === 'Binary_Classification') {
    addLayers(model, ytrain.shape[1], 'sigmoid');
    compileModel(
      model,
      'binaryCrossentropy',
      ['accuracy'],
      optimizer,
      learningRate
    );
    trainLogs = await fitModel(
      batchsize,
      epochs,
      Xtrain,
      ytrain,
      XVal,
      yVal,
      model,
      elementA,
      type,
      elementB,

      ['acc', 'val_acc']
    );
    result = evaluateModel(XTest, yTest, model);
    // sessionStorage.setItem(
    //   'Losses',
    //   JSON.stringify({
    //     trainLoss: trainLogs[99].loss,
    //     valLoss: trainLogs[99].val_loss,
    //     TestLoss: result,
    //     trainAcc: trainLogs[99].acc,
    //     ValAcc: trainLogs[99].val_acc,
    //   })
    // );
  } else if (type === 'MultiClass_Classification') {
    console.log(ytrain.shape[1]);
    addLayers(model, ytrain.shape[1], 'softmax');
    compileModel(
      model,
      'categoricalCrossentropy',
      ['accuracy'],
      optimizer,
      learningRate
    );
    trainLogs = await fitModel(
      batchsize,
      epochs,
      Xtrain,
      ytrain,
      XVal,
      yVal,
      model,
      elementA,
      type,
      elementB,
      ['val_acc', 'acc']
    );
    result = evaluateModel(XTest, yTest, model);
    console.log('hero');
    console.log(trainLogs);
  } else {
    addLayers(model, ytrain.shape[1], 'linear');
    compileModel(
      model,
      'meanSquaredError',
      [r2Score, 'mse'],
      optimizer,
      learningRate
    ); // a revoir
    console.log('here');
    console.log(Xtrain.shape[0]);
    console.log(ytrain.shape[0]);
    console.log(XVal.shape[0]);
    console.log(yVal.shape[0]);
    trainLogs = await fitModel(
      batchsize,
      epochs,
      Xtrain,
      ytrain,
      XVal,
      yVal,
      model,
      elementA,
      type
    );
    console.log(trainLogs);
    result = evaluateModel(XTest, yTest, model);
    // predictXTest(model, XTest, yTest, );
    console.log('heeere1');
    // sessionStorage.setItem(
    //   'Losses',
    //   JSON.stringify({
    //     trainLoss: trainLogs[99].loss,
    //     valLoss: trainLogs[99].val_loss,
    //     TestLoss: result,
    //   })
    // );
    console.log('heeere2');
  }

  // model.summary();
  tfvis.show.modelSummary({ name: 'Model Summary' }, model);
  return { Model: model, Losses: trainLogs[99], TestEvaluation: result };
};
// add Layers
function addLayers(model, shape, activation) {
  // model.add(tf.layers.dense({ units: 64, activation: 'relu' }));
  // model.add(tf.layers.dense({ units: 8, activation: 'relu' }));

  model.add(tf.layers.dense({ units: shape, activation: activation }));
}
// compile Model
function compileModel(model, loss, metrics, optimizer, learningRate) {
  if (optimizer === 'adam') {
    optimizer = tf.train.sgd(learningRate);
  } else {
    optimizer = tf.train.sgd(learningRate);
  }

  //rember meterics should be an array
  model.compile({
    optimizer: optimizer,
    loss: loss,
    metrics: metrics,
  });
}
// fit Model
async function fitModel(
  batchsize,
  epochs,
  Xtrain,
  ytrain,
  XVal,
  yVal,
  model,
  elementA,
  type,
  elementB,
  tabB
) {
  console.log(type);
  const trainLogs = [];
  await model.fit(Xtrain, ytrain, {
    batchSize: batchsize,
    epochs: epochs,
    validationData: [XVal, yVal],
    shuffle: true,

    callbacks: {
      onEpochEnd: async (epoch, logs) => {
        trainLogs.push(logs);
        // console.log(trainLogs);
        tfvis.show.history(elementA, trainLogs, ['loss', 'val_loss']);
        if (type !== 'Regression') {
          tfvis.show.history(elementB, trainLogs, tabB);
        }
      },
    },
  });
  return trainLogs;
}

// evaluate model
function evaluateModel(Xtest, ytest, model) {
  const result = model.evaluate(Xtest, ytest, { batchSize: 4 });
  console.log('losses:', result[0].dataSync()[0]);
  console.log(result[1].dataSync()[0]);
  return result[0].dataSync()[0];
}
// save Model
export async function saveModel(model, modelName, x, y) {
  await model.save('localstorage://' + modelName);

  let xLabelsSerialized = JSON.stringify(x);
  let yLabelsSerialized = JSON.stringify(y);
  let path = `${modelName}/xLables`;
  let pathy = `${modelName}/yLabel`;
  localStorage.setItem(path, xLabelsSerialized);
  localStorage.setItem(pathy, yLabelsSerialized);
}
// predict
export async function predictResult(modelName, tab) {
  let x = JSON.parse(localStorage.getItem(`${modelName}/xLables`));
  for (let i = 0; i < tab.length; i++) {
    if (x[i].type === 'float32') {
      parseFloat(tab[i]);

      tab[i] = (tab[i] - x[i].minSacle) / (x[i].maxScale - x[i].minSacle);
    } else if (x[i].type === 'int32') {
      parseInt(tab[i]);

      tab[i] = (tab[i] - x[i].minSacle) / (x[i].maxScale - x[i].minSacle);
    } else if (x[i].type === 'String') {
      let labels = x[i].labels;
      let l = labels[tab[i]];
      tab[i] = l;

      tab[i] = (tab[i] - x[i].minSacle) / (x[i].maxScale - x[i].minSacle);
    } else if (x[i].type === 'date') {
      // when using the month-day encoding
      let min = new Date(`${tab[i].split('-')[0]}-01-01`);
      min = new Date(min);
      let datei = new Date(tab[i]);
      // with handle date : let min = new Date(x[i].min);
      let numberofday = (datei - min) / (1000 * 60 * 60 * 24);
      console.log(numberofday);
      parseInt(numberofday);

      tab[i] = (numberofday - x[i].minSacle) / (x[i].maxScale - x[i].minSacle);

      console.log(tab[i]);
    }
  }

  console.log(tab);
  const model = await tf.loadLayersModel(`localstorage://${modelName}`);

  let a = model.predict(tf.tensor(tab, [1, tab.length]));
  a.print();
  let y = JSON.parse(localStorage.getItem(`${modelName}/yLabel`));
  if (y.learningType === 'MultiClass_Classification') {
    var pIndex = tf.argMax(a, 1).dataSync();

    console.log(y.labels[pIndex]);
    let retour = {
      name: JSON.parse(localStorage.getItem(`${modelName}/yLabel`)).name,
      label: y.labels[pIndex],
      proba: (a.dataSync()[pIndex] * 100).toFixed(2),
    };
    console.log(retour);
    return retour;
  } else {
    // (a_mean)/std
    const mean = tf.scalar(y.mean);
    const std = tf.scalar(y.std);

    let pred = a.mul(std).add(mean);
    pred.print();
    // if (
    //   JSON.parse(localStorage.getItem(`${modelName}/yLabel`)).type === 'int32'
    // ) {
    //   pred = pred.toFixed(0);
    // } else {
    //   pred = pred.toFixed(2);
    // }
    let retour = {
      name: JSON.parse(localStorage.getItem(`${modelName}/yLabel`)).name,
      prediction: pred.dataSync(),
    };
    return retour;
  }
}
function r2Score(yTrue, yPred) {
  const yMean = tf.mean(yTrue);
  const totalSumOfSquares = tf.sum(tf.square(tf.sub(yTrue, yMean)));
  const residualSumOfSquares = tf.sum(tf.square(tf.sub(yTrue, yPred)));
  const r2 = tf.sub(1.0, tf.div(residualSumOfSquares, totalSumOfSquares));
  return r2;
}
