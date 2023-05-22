import React from 'react';
import { green } from '@mui/material/colors';
import { Button, Typography } from '@mui/material';
import { Box } from '@mui/material';
import { TextField } from '@mui/material';
import { predictResult } from '../utils/train';
import List from '@mui/material/List';
import ListItem from '@mui/material/ListItem';
import ListItemText from '@mui/material/ListItemText';

function Predict({ x, y, modelName }) {
  const [newData, setNewData] = React.useState();
  const [result, setResult] = React.useState();

  async function handlePredict(e) {
    console.log(newData);
    let arr = [];
    for (let i = 0; i < x.length; i++) {
      arr.push(newData[x[i].name]);
    }
    console.log(arr);

    setResult(await predictResult(modelName, arr));
  }

  return (
    <Box>
      <Typography sx={{ marginTop: 5, marginRight: 7, marginBottom: 2 }}>
        Predict Result :{' '}
      </Typography>

      {x.map((row) => (
        <TextField
          label={row.name}
          name={row.name}
          // value={newData[row.name]}
          variant="outlined"
          fullWidth
          required
          sx={{ marginTop: 3 }}
          onChange={(e) => {
            const { name, value } = e.target;
            setNewData({ ...newData, [name]: value });
          }}
        />
      ))}
      <Button
        onClick={handlePredict}
        type="submit"
        variant="outlined"
        color="primary"
        sx={{ marginLeft: 2, marginTop: 2, marginBottom: 2 }}
      >
        Predict
      </Button>
      {result &&
        JSON.parse(localStorage.getItem(`${modelName}/yLabel`)).learningType ===
          'MultiClass_Classification' && (
          <List sx={{ width: '100%', maxWidth: 360, bgcolor: green[500] }}>
            <ListItem>
              <ListItemText
                primary={result.name + ':' + result.label}
                secondary={result.proba + '%'}
              />
            </ListItem>
          </List>
        )}
      {result &&
        JSON.parse(localStorage.getItem(`${modelName}/yLabel`)).learningType ===
          'Regression' && (
          <List sx={{ width: '100%', maxWidth: 360, bgcolor: green[500] }}>
            <ListItem>
              <ListItemText
                primary={' Regression :' + result.name}
                secondary={result.prediction}
              />
            </ListItem>
          </List>
        )}
    </Box>
  );
}

export default Predict;
