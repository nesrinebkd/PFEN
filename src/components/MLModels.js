import React, { useEffect } from 'react';
import { Box } from '@mui/material';
import Table from '@mui/material/Table';
import TableBody from '@mui/material/TableBody';
import TableCell from '@mui/material/TableCell';
import TableContainer from '@mui/material/TableContainer';
import TableHead from '@mui/material/TableHead';
import TableRow from '@mui/material/TableRow';
import Paper from '@mui/material/Paper';
import Button from '@mui/material/Button';
import Predict from './Predict';
function MLModels() {
  const [x, setX] = React.useState();
  const [y, setY] = React.useState();
  const [modelName, setModelName] = React.useState();
  const [models, setModels] = React.useState([]);
  const [isChanged, setIsChanged] = React.useState(false);
  useEffect(() => {
    for (let i = 0; i < localStorage.length; i++) {
      const key = localStorage.key(i);

      if (
        key.startsWith('tensorflowjs_models') &&
        !models.includes(key.split('/')[1])
      )
        setModels([...models, key.split('/')[1]]);
    }
  }, [models]);
  return (
    <Box>
      {models}
      <TableContainer component={Paper}>
        <Table sx={{ minWidth: 650 }} aria-label="simple table">
          <TableHead>
            <TableRow>
              <TableCell component="th" scope="row">
                Model Name
              </TableCell>
              <TableCell component="th" scope="row">
                Last Trained
              </TableCell>
              <TableCell component="th" scope="row">
                Type
              </TableCell>
              <TableCell component="th" scope="row">
                Actions
              </TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {models.map((row) => (
              <TableRow
                key={row}
                sx={{ '&:last-child td, &:last-child th': { border: 0 } }}
              >
                <TableCell>{row}</TableCell>
                <TableCell>
                  {
                    JSON.parse(
                      localStorage.getItem(`tensorflowjs_models/${row}/info`)
                    ).dateSaved.split('T')[0]
                  }
                </TableCell>
                <TableCell>
                  {
                    JSON.parse(localStorage.getItem(`${row}/yLabel`))
                      .learningType
                  }
                </TableCell>
                <TableCell>
                  <Button
                    value={row}
                    variant="outlined"
                    color="primary"
                    sx={{ marginLeft: 0.5, marginRight: 0.5 }}
                    onClick={(e) => {
                      // sessionStorage.removeItem(`tensorflowjs_models/${row}/yLabel`)
                    }}
                  >
                    Show Details
                  </Button>
                  <Button
                    value={row}
                    variant="outlined"
                    color="primary"
                    sx={{ marginLeft: 0.5, marginRight: 0.5 }}
                    onClick={(e) => {
                      // sessionStorage.removeItem(`tensorflowjs_models/${row}/yLabel`)
                    }}
                  >
                    Delete
                  </Button>
                  <Button
                    value={row}
                    variant="outlined"
                    color="primary"
                    sx={{ marginLeft: 0.5, marginTop: 0.5 }}
                    onClick={(e) => {
                      console.log(e.target.value);
                      setIsChanged(true);
                      setModelName(e.target.value);
                      setY(
                        JSON.parse(
                          localStorage.getItem(`${e.target.value}/yLabel`)
                        )
                      );
                      setX(
                        JSON.parse(
                          localStorage.getItem(`${e.target.value}/xLables`)
                        )
                      );
                    }}
                  >
                    Predict
                  </Button>
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </TableContainer>
      {isChanged && <Predict x={x} y={y} modelName={modelName}></Predict>}
    </Box>
  );
}

export default MLModels;
