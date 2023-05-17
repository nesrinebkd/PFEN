import React, { useEffect } from 'react';
import Box from '@mui/material/Box';
import { useRef } from 'react';
import { TextField } from '@mui/material';
// import Container from '@mui/material/Container';
import FormControl from '@mui/material/FormControl';
import FormControlLabel from '@mui/material/FormControlLabel';
import FormGroup from '@mui/material/FormGroup';
import InputLabel from '@mui/material/InputLabel';
import MenuItem from '@mui/material/MenuItem';
import Select from '@mui/material/Select';
import Table from '@mui/material/Table';
import TableBody from '@mui/material/TableBody';
import TableCell from '@mui/material/TableCell';
import TableContainer from '@mui/material/TableContainer';
import TableHead from '@mui/material/TableHead';
import TableRow from '@mui/material/TableRow';
import Paper from '@mui/material/Paper';
import Stepper from '@mui/material/Stepper';
import Step from '@mui/material/Step';
import StepLabel from '@mui/material/StepLabel';
import Checkbox from '@mui/material/Checkbox';
import Button from '@mui/material/Button';
import Typography from '@mui/material/Typography';
import { cleanData } from '../utils/clean';
import { createModel } from '../utils/train';
import { saveModel } from '../utils/train';
import MLModels from './MLModels';
import Predict from './Predict';
const steps = [
  'Select a column to predict and data to study',
  'Choose a model',
  'Name and train',
];

function BuildModels({ Schema, data }) {
  const [activeStep, setActiveStep] = React.useState(0);
  const schema = Schema.columns;
  const [y, setY] = React.useState('');
  const [x, setX] = React.useState('');
  const [yError, setyError] = React.useState(false);
  const [isCustomized, setIsCustomized] = React.useState(false);
  const [model, setModel] = React.useState('');
  const [result, setResult] = React.useState({});
  const [trained, setIstrained] = React.useState(false);
  const [finishTrainedProcess, setFinishTrainedProcess] = React.useState({});
  const historyRef1 = useRef(null);
  const historyRef2 = useRef(null);
  // const [Data, setData] = React.useState(data);

  let i = 0;
  const labels = Object.keys(schema);

  const handleNext = async () => {
    if (activeStep === 0) {
      setyError(false);

      console.log(x);
      console.log(y);
      if (y === '') {
        setyError(true);
        alert('fill y first');
      } else if (x === '') {
        alert('fill x first');
      } else {
        let res = await cleanData(data, x, y);

        setResult(res);
        console.log(res);
        setX(res.xcolumns);
        setY(res.ylabel);
        i = 0;
        setActiveStep((prevActiveStep) => prevActiveStep + 1);
      }
    } else if (activeStep === 1) {
      await setActiveStep((prevActiveStep) => prevActiveStep + 1);
      console.log(y.learningType);
      let trainProcess = await createModel(
        result.Xtrain,
        result.ytrain,
        result.XVal,
        result.YVal,
        result.XTest,
        result.YTest,
        y.learningType,
        historyRef1.current,
        historyRef2.current
      );
      setFinishTrainedProcess(trainProcess);
      setIstrained(true);
    } else if (activeStep === 2) {
      if (model === '') {
        alert('you should give a name to your model ! ');
      } else {
        console.log('we are here ');
        setActiveStep((prevActiveStep) => prevActiveStep + 1);
        await saveModel(finishTrainedProcess.Model, model, x, y);
      }
    } else {
      setActiveStep((prevActiveStep) => prevActiveStep + 1);
    }
  };

  const handleBack = () => {
    if (activeStep === 1) {
      setY('');
      setX([]);
      // setData(data);
    }
    setActiveStep((prevActiveStep) => prevActiveStep - 1);
  };

  const handleReset = () => {
    setActiveStep(0);
  };

  const getLables = (labels) => {
    return labels.map((label) => (
      <MenuItem value={label} key={label}>
        {/* {label + ':' + schema[label].type} */}
        {label}
      </MenuItem>
    ));
  };

  const getLablesXceptY = (labels) => {
    return labels.map((label) =>
      label !== y.name ? (
        <FormControlLabel
          control={<Checkbox value={label} onChange={handleChange} />}
          label={label}
        />
      ) : (
        <FormControlLabel
          control={<Checkbox value={label} disabled />}
          label={label}
        />
      )
    );
  };

  const handleChange = (e) => {
    const value = e.target.value;
    const checked = e.target.checked;

    if (checked) {
      console.log(value);
      console.log(checked);
      setX([...x, { name: value, type: schema[e.target.value].type }]);
    } else {
      setX(x.filter((e) => e.name !== value));
    }
  };

  const handleCustomize = () => {
    setIsCustomized(true);
  };
  return (
    <Box sx={{ width: '100%' }}>
      <Stepper activeStep={activeStep}>
        {steps.map((label, index) => {
          const stepProps = {};
          const labelProps = {};

          return (
            <Step key={label} {...stepProps}>
              <StepLabel {...labelProps}>{label}</StepLabel>
            </Step>
          );
        })}
      </Stepper>
      {activeStep === steps.length ? (
        <React.Fragment>
          <Typography sx={{ mt: 2, mb: 1 }}>
            All steps completed - you&apos;re finished
          </Typography>
          <Predict x={x} y={y} modelName={model}></Predict>
          <Box sx={{ display: 'flex', flexDirection: 'row', pt: 2 }}>
            <Box sx={{ flex: '1 1 auto' }} />
            <Button variant="outlined" onClick={handleReset}>
              Reset
            </Button>
          </Box>
        </React.Fragment>
      ) : (
        <React.Fragment>
          <Box sx={{ mt: 8 }}>
            {activeStep === 0 && (
              <Box>
                <Typography sx={{ mt: 2, mb: 1, fontsize: 1 }}>
                  Step 1 :What do you want to predict ?
                </Typography>
                <Typography sx={{ ml: 3, mt: 2, mb: 1, fontSize: 15 }}>
                  Select the outcome column you'd like to make predictions
                  about, so we can recommand the best model .
                </Typography>
                <FormControl
                  variant="filled"
                  fullWidth
                  size="string"
                  sx={{ fontSize: 10, size: 'string' }}
                >
                  <InputLabel
                    id="demo-simple-select-label"
                    sx={{ mb: 1, fontSize: 15 }}
                  >
                    Label Y :{' '}
                  </InputLabel>
                  <Select
                    labelId="demo-simple-select-label"
                    id="demo-simple-select"
                    value={y.name}
                    label="y"
                    error={yError}
                    onChange={(e) =>
                      setY({
                        name: e.target.value,
                        type: schema[e.target.value].type,
                      })
                    }
                  >
                    {getLables(labels)}
                  </Select>
                </FormControl>
                <Typography sx={{ mt: 2, mb: 1, fontSize: 18 }}>
                  Step 2:select the data your model should study ?
                </Typography>
                <FormGroup>{getLablesXceptY(labels)}</FormGroup>
              </Box>
            )}
            {isCustomized === false && activeStep === 1 && (
              <Box>
                <Typography sx={{ mb: 1 }}>
                  These are the X features you selected{' '}
                </Typography>
                <TableContainer component={Paper}>
                  <Table sx={{ minWidth: 650 }} aria-label="simple table">
                    <TableHead>
                      <TableRow>
                        <TableCell component="th" scope="row">
                          Feature
                        </TableCell>
                        <TableCell component="th" scope="row">
                          Name
                        </TableCell>
                        <TableCell component="th" scope="row">
                          Type
                        </TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {x.map((row) => (
                        <TableRow
                          key={row.name}
                          sx={{
                            '&:last-child td, &:last-child th': { border: 0 },
                          }}
                        >
                          <TableCell component="th" scope="row">
                            {i++}
                          </TableCell>
                          <TableCell>{row.name}</TableCell>
                          <TableCell>{row.type}</TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </TableContainer>
                <Typography sx={{ mt: 2, mb: 1 }}>
                  Based on the column you selected as an output, we recommand a{' '}
                  {y.learningType} model .
                </Typography>
                <Box textAlign="center">
                  <Button
                    onClick={handleCustomize}
                    sx={{ mr: 1, mt: 2 }}
                    variant="outlined"
                  >
                    If you want to customize the neural network press here{' '}
                  </Button>
                </Box>
              </Box>
            )}
            {isCustomized === true && activeStep === 1 && <Box>hello</Box>}
            {activeStep === 2 && (
              <Box>
                <Box>
                  <Box ref={historyRef1}></Box>
                  <Box ref={historyRef2}></Box>
                </Box>
                <Typography>Name :</Typography>
                <TextField
                  label="Name the Model"
                  variant="outlined"
                  fullWidth
                  required
                  sx={{ marginTop: 3 }}
                  onChange={(e) => setModel(e.target.value)}
                />
              </Box>
            )}
            {trained && activeStep === 2 && (
              <Box>
                <br />
                <Typography>Losses : </Typography>
                <br />
                <TableContainer component={Paper}>
                  <Table sx={{ minWidth: 650 }} aria-label="simple table">
                    <TableHead>
                      <TableRow>
                        <TableCell component="th" scope="row">
                          Loss
                        </TableCell>
                        <TableCell component="th" scope="row">
                          Value
                        </TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {Object.keys(finishTrainedProcess.Losses).map((row) => (
                        <TableRow
                          key={row.name}
                          sx={{
                            '&:last-child td, &:last-child th': { border: 0 },
                          }}
                        >
                          <TableCell component="th" scope="row">
                            {row}
                          </TableCell>
                          <TableCell>
                            {finishTrainedProcess.Losses[row]}
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </TableContainer>
                <Typography>
                  The evaluaton on test set gave :{' '}
                  {finishTrainedProcess.TestEvaluation}
                </Typography>
              </Box>
            )}
            {/* {activeStep === 3 && <MLModels></MLModels>} */}
          </Box>

          <Box sx={{ display: 'flex', flexDirection: 'row', pt: 2 }}>
            <Button
              color="inherit"
              disabled={activeStep === 0}
              onClick={handleBack}
              sx={{ mr: 1 }}
              variant="outlined"
            >
              Back
            </Button>
            <Box sx={{ flex: '1 1 auto' }} />
            <Button onClick={handleNext} variant="outlined">
              {activeStep === steps.length - 1
                ? 'Finish'
                : activeStep === 1
                ? 'Train'
                : 'Next'}
            </Button>
          </Box>
        </React.Fragment>
      )}
    </Box>
  );
}

export default BuildModels;
