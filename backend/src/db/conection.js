require('dotenv').config()
const mongoose = require('mongoose');

mongoose.connect(process.env.DB_URL, {
  useNewUrlParser:true,
}).then(()=>{
  console.log("connection succesful")
}).catch((e)=>{
  console.log("No connection with Database !!!" )
  console.log(e)
})
