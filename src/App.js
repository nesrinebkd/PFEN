import './App.css';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { createTheme, ThemeProvider } from '@mui/material/styles';
import { green } from '@mui/material/colors';
import TheRoot from './components/TheRoot';
function App() {
  const theme = createTheme({
    palette: {
      primary: {
        main: green[500],
      },
    },
    typography: {
      fontFamily: 'Quicksand',
      fontWeightLight: 400,
      fontWeightMedium: 500,
      fontWeightBold: 700,
      fontWeightRegular: 600,
    },
  });
  return (
    <ThemeProvider theme={theme}>
      <Router>
        <Routes>
          <Route path="/" element={<TheRoot />}></Route>
        </Routes>
      </Router>
    </ThemeProvider>
  );
}

export default App;
