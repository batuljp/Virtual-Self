const express = require("express");
const cors = require("cors");
const path = require("path");

const app = express();

const ipaddress = "0.0.0.0";
const port = 19019;

app.use(cors());

app.use("/", express.static(path.join(__dirname, "outputs")));

app.listen(port, ipaddress, () =>
  console.log(`Listening at ${ipaddress}:${port}...`)
);

// Path : 68.183.80.142:19019/final.mp4