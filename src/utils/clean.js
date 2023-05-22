import * as dfd from 'danfojs';
import * as tf from '@tensorflow/tfjs';

export const cleanData = async (data, x, y, learningtyp = '') => {
  //shuffleData

  console.log(x, y, learningtyp);
  const df = new dfd.DataFrame(data);
  // Extract the needed data for the learning to remove the null rows
  // 1- put x and y in one array columns
  let columns = [];
  for (let i = 0; i < x.length; i++) {
    columns.push(x[i]);
  }
  columns.push(y);
  console.log(columns);
  //2- get the data and remove the null rows
  const newDf = dataNeeded(df, columns);
  let df_drop = newDf.dropNa({ axis: 1 });

  // get Xtrain and ytrain
  let Xtrain, ytrain;
  let Y = [];
  Y.push(y);
  ytrain = dataNeeded(df_drop, Y);

  //get the learning type
  let type;
  let serie = ytrain.nUnique(0);
  let a = serie.values[0];
  if (learningtyp === 'Binary Classification' || a === 2) {
    //label encoding
    type = 'B_Classification';
    let ylabel = labelEncoding(ytrain, y.name);
    y = {
      name: y.name,
      type: 'String',
      labels: ylabel,
      learningType: 'Binary_Classification',
    };
  } else if (
    learningtyp === 'Multi-class Classification' ||
    (learningtyp === '' &&
      ((a > 2 && a < 12) || ytrain.ctypes.values[0] === 'string'))
  ) {
    console.log('Multi-class Classification');
    type = 'MultiClass_Classification';
    // json
    let dfDropJson = dfd.toJSON(df_drop);
    console.log(dfDropJson);
    //shuffle data
    dfDropJson = shuffle(dfDropJson);
    df_drop = new dfd.DataFrame(dfDropJson);
    ytrain = dataNeeded(df_drop, Y);
    let oneHotResult = oneHotEncodingX(ytrain, [y.name]);
    ytrain = oneHotResult.dum_df;

    y = {
      name: y.name,
      type: 'float',
      learningType: type,
      labels: oneHotResult.labels,
    };
  } else if (
    learningtyp === 'Regression' ||
    (learningtyp === '' &&
      (ytrain.ctypes.values[0] === 'int32' ||
        ytrain.ctypes.values[0] === 'float32'))
  ) {
    console.log('regression here');
    type = 'Regression';
    let arr = [];
    for (let i = 0; i < x.length; i++) {
      arr.push(x[i].name);
    }
    //group data to avoid redundunt rows with diff outputs in regression model
    let grp = df_drop.groupby(arr);
    df_drop = grp.agg({ [y.name]: 'sum' });

    df_drop.print();
    Y[0].name = [y.name] + '_sum';
    // json
    let dfDropJson = dfd.toJSON(df_drop);
    console.log(dfDropJson);
    //shuffle data
    dfDropJson = shuffle(dfDropJson);
    df_drop = new dfd.DataFrame(dfDropJson);
    console.log('data after grouping and shuffling ');
    df_drop.print();
    ytrain = dataNeeded(df_drop, Y);
    ytrain.print();
  }
  //prepare the new X and y train

  Xtrain = dataNeeded(df_drop, x);
  // let newXtrain = await handleDates(dfd.toJSON(Xtrain), x);

  let newXtrain = await intCodMonthDay(dfd.toJSON(Xtrain), x);
  console.log(newXtrain.result);
  Xtrain = new dfd.DataFrame(newXtrain.result);
  Xtrain.print();
  for (let i = 0; i < x.length; i++) {
    if (x[i].type === 'date') {
      Xtrain.drop({ columns: [x[i].name], inplace: true });
    }
  }
  Xtrain.print();
  // ytrain = dataNeeded(df_drop, Y);
  // labelEncoding for Xtrain
  let X = handleDatesAndEncoding(Xtrain);
  Xtrain = X.df;
  let tString = X.t;

  // split the data :
  // 60% training
  let indexVal = Xtrain.shape[0] * 0.8;
  let subXtrain = Xtrain.iloc({ rows: [` 0: ${indexVal}`] });
  let subYtrain = ytrain.iloc({ rows: [` 0: ${indexVal}`] });

  // 20% validation
  let indexTest = Xtrain.shape[0] * 0.1 + indexVal;
  let Xval = Xtrain.iloc({ rows: [` ${indexVal}: ${indexTest}`] });
  let Yval = ytrain.iloc({ rows: [` ${indexVal}: ${indexTest}`] });

  // 20% Test
  let XTest = Xtrain.iloc({ rows: [` ${indexTest}:`] });
  let YTest = ytrain.iloc({ rows: [` ${indexTest}:`] });

  // Normalization of X and Y train
  let tabs = getMinMaxX(subXtrain);
  let NormlizedX = normlize(subXtrain, Xval, XTest);
  subXtrain = NormlizedX.xtrain;
  Xval = NormlizedX.vlaX;
  XTest = NormlizedX.testX;

  // Normalizing ytrain , yval and y test  if the learning type is regression
  if (type === 'Regression') {
    getSkewnessKurtosis(subYtrain);

    const meanStd = getMeanStd(subYtrain);
    // This must be implemented in a function ...
    let scaler = new dfd.StandardScaler();
    scaler.fit(subYtrain);
    subYtrain = scaler.transform(subYtrain);
    Yval = scaler.transform(Yval);
    YTest = scaler.transform(YTest);
    y.name = y.name.split('_sum')[0];
    y = {
      name: y.name,
      type: ytrain.ctypes.values[0],
      learningType: 'Regression',
      mean: meanStd.mean,
      std: meanStd.std,
    };
  }

  // x , types and labels
  console.log(newXtrain.dateType);
  let xcolumns = getKnowledge(subXtrain, tabs, tString, newXtrain.dateType);

  // Affichage :
  console.log(xcolumns);
  subXtrain.print();
  subYtrain.print();
  Xval.print();
  Yval.print();
  XTest.print();
  YTest.print();
  console.log(subXtrain.shape[0]);
  console.log(subYtrain.shape[0]);
  console.log(Xval.shape[0]);
  console.log(Yval.shape[0]);
  console.log(XTest.shape[0]);
  console.log(YTest.shape[0]);

  return {
    xcolumns: xcolumns,
    ylabel: y,
    // train
    Xtrain: dfd.toJSON(subXtrain),
    ytrain: dfd.toJSON(subYtrain),
    // validation
    XVal: dfd.toJSON(Xval),
    YVal: dfd.toJSON(Yval),
    // Test
    XTest: dfd.toJSON(XTest),
    YTest: dfd.toJSON(YTest),
  };
};

// shuffle data using the Fisher-Yates shuffle algorithm
const shuffle = (sourceArray) => {
  for (var i = 0; i < sourceArray.length - 1; i++) {
    var j = i + Math.floor(Math.random() * (sourceArray.length - i));

    var temp = sourceArray[j];
    sourceArray[j] = sourceArray[i];
    sourceArray[i] = temp;
  }
  return sourceArray;
};

// Extract the needed Data
const dataNeeded = (df, col) => {
  let tabOfNames = getInputs(col);
  let newDf = df.loc({ columns: tabOfNames });
  return newDf;
};
const getInputs = (x) => {
  var tab = [];
  for (let i = 0; i < x.length; i++) {
    tab.push(x[i].name);
  }
  return tab;
};

// Label Encoding
const labelEncoding = (df, col_name) => {
  let encode = new dfd.LabelEncoder();
  encode.fit(df[col_name]);
  // console.log(encode.$labels);
  let sf_enc = encode.transform(df[col_name].values);
  df.addColumn(col_name, sf_enc, { inplace: true });
  return encode.$labels;
};

// OneHotEncoding
const oneHotEncodingX = (df, tab_col_name) => {
  let dum_df = dfd.getDummies(df, { columns: tab_col_name }); // try it with the same name of df !!
  let encode = new dfd.OneHotEncoder();

  encode.fit(df[tab_col_name[0]]);

  let labels = encode.$labels;
  return { dum_df, labels };
};

//LabelEncoding
const handleDatesAndEncoding = (df) => {
  // handle Strings : OneHotEncoding / LabelEncoding : for the moment we are using Labelencoding
  let s = df.ctypes;
  let t = [];
  for (let i = 0; i < s.index.length; i++) {
    if (s.values[i] === 'string') {
      t.push({ name: s.index[i], labels: labelEncoding(df, s.index[i]) });
    }
    // // hada lizdtih
    // if (s.values[i] === 'int32') {
    //   let serie = df.loc({ columns: [s.index[i]] }).nUnique(0);
    //   let a = serie.values[0];
    //   if (a > 2 && a < 12) {
    //     t.push({ name: s.index[i], labels: labelEncoding(df, s.index[i]) });
    //   }
    // }
  }
  return { df, t };
};

//get Min and Max
const getMinMaxX = (df) => {
  let tf_tensor = df.tensor;

  let min = tf_tensor.min(0).dataSync();
  let max = tf_tensor.max(0).dataSync();

  let tabMin = [];
  let tabMax = [];
  for (let i = 0; i < min.length; i++) {
    tabMin[i] = min[i];
    tabMax[i] = max[i];
  }

  return { tabMin, tabMax };
};

// Normalization
const normlize = (X, valX, TestX) => {
  let scaler = new dfd.MinMaxScaler();
  scaler.fit(X);
  let XtrainNormalized = scaler.transform(X);
  let XValNormalized = scaler.transform(valX);
  let XTest = scaler.transform(TestX);

  return { xtrain: XtrainNormalized, vlaX: XValNormalized, testX: XTest };
};

//getMeanandStd for regression
const getMeanStd = (df) => {
  let tf_tensor = df.tensor;
  console.log('df');
  tf_tensor.print();
  let mean = tf_tensor.mean(0).dataSync();
  //  // std

  //     x-mean[0]
  mean = tf.scalar(mean[0]);
  //     std : standard diviation
  let std = tf_tensor
    .sub(mean)
    .square()
    .sum(0)
    .div(tf.scalar(df.shape[0]))
    .sqrt();

  return { std: std.dataSync()[0], mean: mean.dataSync()[0] };
};
// to get all features then load it to the local storage
const getKnowledge = (df, tabs, tString, x) => {
  let serie = df.ctypes;
  console.log(serie);
  let ob = [];
  for (let i = 0; i < serie.index.length; i++) {
    ob.push({
      name: serie.index[i],
      type: serie.values[i],
      minSacle: tabs.tabMin[i],
      maxScale: tabs.tabMax[i],
    });
  }
  for (let j = 0; j < tString.length; j++) {
    for (let k = 0; k < ob.length; k++) {
      if (ob[k].name === tString[j].name) {
        ob[k].type = 'String';
        ob[k].labels = tString[j].labels;
      }
    }
  }

  for (let k = 0; k < ob.length; k++) {
    if (ob[k].name.endsWith('_grp')) {
      for (let j = 0; j < x.length; j++) {
        if (ob[k].name.startsWith(x[j].name)) {
          // x here is a dateType table
          ob[k].type = 'date';
          ob[k].min = x[j].min;
        }
      }
    }
  }

  return ob;
};

const handleDates = async (data, x) => {
  let dataDate = {};
  let newData = [];
  let dateType = [];

  for (let i = 0; i < data.length; i++) {
    newData.push({});
  }
  for (let i = 0; i < x.length; i++) {
    if (x[i].type === 'date') {
      dataDate[x[i].name] = [];
      for (let j = 0; j < data.length; j++) {
        let k = new Date(data[j][x[i].name]);
        // console.log(k.toISOString().slice(0, 10));
        dataDate[x[i].name].push(k.toISOString().slice(0, 10));
      }
      const minTimestamp = Math.min(
        ...dataDate[x[i].name].map((date) => Date.parse(date))
      );
      let min = new Date(minTimestamp).toISOString().split('T')[0];
      console.log(min);

      dateType.push({ name: x[i].name, min: min });

      for (let k = 0; k < data.length; k++) {
        let datei = new Date(data[k][x[i].name]);
        min = new Date(min);
        newData.push();
        newData[k][x[i].name + '_grp'] = (datei - min) / (1000 * 60 * 60 * 24);
      }
    }
  }
  console.log(newData);
  console.log(dateType);
  // concatenate
  var result = [];
  for (let i = 0; i < data.length; i++) {
    result[i] = { ...data[i], ...newData[i] };
  }

  return { result, dateType };
};

const getSkewnessKurtosis = (df) => {
  let meanStd = getMeanStd(df);
  let mean = meanStd.mean;
  let std = meanStd.std;
  console.log(mean, std);
  let tf_tensor = df.tensor;
  console.log('df');
  tf_tensor.print();
  //  // std

  //     x-mean[0]
  mean = tf.scalar(mean);
  //     skewness
  let skewness = tf_tensor
    .sub(mean)
    .div(tf.scalar(std))
    .pow(3)
    .sum(0)
    .div(tf.scalar(df.shape[0]));
  let kurtosis = tf_tensor
    .sub(mean)
    .div(tf.scalar(std))
    .pow(4)
    .sum(0)
    .div(tf.scalar(df.shape[0]));

  kurtosis.print();
  skewness.print();

  if (skewness.dataSync() > 1 || skewness.dataSync() < -1) {
    alert('Warning : your data is Asymetric , this can lead to poor results');
  }
};

//

const intCodMonthDay = async (data, x) => {
  let dataDate = {};
  let newData = [];
  let dateType = [];

  for (let i = 0; i < data.length; i++) {
    newData.push({});
  }
  for (let i = 0; i < x.length; i++) {
    if (x[i].type === 'date') {
      dataDate[x[i].name] = [];
      for (let j = 0; j < data.length; j++) {
        let k = new Date(data[j][x[i].name]);
        // console.log(k.toISOString().slice(0, 10));
        dataDate[x[i].name].push(k.toISOString().slice(0, 10));
      }

      dateType.push({ name: x[i].name, min: '2000-01-01' });

      for (let k = 0; k < data.length; k++) {
        var min = new Date(`${data[k][x[i].name].split('-')[0]}-01-01`);
        var date = new Date(data[k][x[i].name]);
        var mini = new Date(min);
        newData[k][x[i].name + '_grp'] = (date - mini) / (1000 * 60 * 60 * 24);
      }
    }
  }
  console.log(newData);
  console.log(dateType);
  // concatenate
  var result = [];
  for (let i = 0; i < data.length; i++) {
    result[i] = { ...data[i], ...newData[i] };
  }

  return { result, dateType };
};

const splitDate = async (data, x) => {};
