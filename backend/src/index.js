require('dotenv').config()
const cors = require('cors');
const express = require("express")
const app = express()
const router = new express.Router();
const bcrypt = require("bcryptjs")
const multer = require("multer")
const cookieParser = require('cookie-parser')
const nodemailer = require('nodemailer');
const randomstring = require('randomstring');
require("../src/db/conection.js");
const socket = require("socket.io");

const port = process.env.PORT || 5000

const User = require("../src/models/users")
const Messages = require("../src/models/message.js");
const Post = require('../src/models/posts.js');


app.use(express.json())
app.use(express.urlencoded({
  extended: false
}))
app.use(cors({
  origin: "*"
}))

app.use(cookieParser())


// user registration and login
// Registration Endpoint
app.post('/register', async (req, res) => {
  const { name, email, password } = req.body;

  // Check if user with the same email already exists
  const existingUser = await User.findOne({ email });
  if (existingUser) {
    return res.status(400).json({ message: 'User with this email already exists' });
  }

  // Hash the password
  const hashedPassword = await bcrypt.hash(password, 10);

  // Create a new user
  const newUser = new User({
    name,
    email,
    password: hashedPassword,
  });

  // Save the user to the database
  try {
    await newUser.save();
    res.status(200).json({ message: 'Registration successful' });
  } catch (error) {
    res.status(500).json({ message: 'Registration failed' });
  }
});

// Login Endpoint
app.post('/login', async (req, res) => {
  const { email, password } = req.body;

  // Find the user by email
  const user = await User.findOne({ email });

  // If user not found or password is incorrect
  if (!user || !(await bcrypt.compare(password, user.password))) {
    return res.status(401).json({ message: 'Invalid email or password' });
  }

  // Successful login
  res.status(200).json({ message: 'Login successful', userData: user });
});



// Generate OTP
const generateOTP = () => {
  return randomstring.generate({
    length: 6,
    charset: 'numeric'
  });
};

// Nodemailer transporter setup
const transporter = nodemailer.createTransport({
  service: 'Gmail',
  auth: {
    user: 'ncddocverify@gmail.com',
    pass: 'xxkw ndrt kfrj xxiy',
  }
});

// Endpoint to initiate forgot password process
app.post('/forgot-password', async (req, res) => {
  const { email } = req.body;
  console.log(email)

  try {
    const user = await User.findOne({ email });
    if (!user) {
      return res.status(404).json({ message: 'User not found' });
    }

    const otp = generateOTP();
    user.resetPasswordOTP = otp;
    await user.save();

    const mailOptions = {
      from: 'ncddocverify@gmail.com',
      to: email,
      subject: 'Reset Password OTP',
      text: `Your OTP to reset the password is: ${otp}`
    };

    transporter.sendMail(mailOptions, (error, info) => {
      if (error) {
        console.log(error);
        res.status(500).json({ message: 'Failed to send OTP' });
      } else {
        console.log('Email sent: ' + info.response);
        res.status(200).json({ message: 'OTP sent successfully' });
      }
    });
  } catch (error) {
    console.log(error);
    res.status(500).json({ message: 'Internal server error' });
  }
});

// Endpoint to verify OTP and reset password
app.post('/reset-password', async (req, res) => {
  const { email, otp, newPassword } = req.body;



  try {
    const user = await User.findOne({ email });
    console.log(`sended otp : ${otp}`)
    console.log(`generated otp : ${user.resetPasswordOTP}`)

    if (!user || user.resetPasswordOTP !== otp) {
      return res.status(400).json({ message: 'Invalid OTP' });
    }

    // Hash the new password
    const hashedPassword = await bcrypt.hash(newPassword, 10);
    user.password = hashedPassword;
    user.resetPasswordOTP = null; // Clear OTP after successful reset
    await user.save();

    res.status(200).json({ message: 'Password reset successful' });
  } catch (error) {
    console.log(error);
    res.status(500).json({ message: 'Internal server error' });
  }
});
// user registration and login





//endpoint to access all the users except the user who's is currently logged in!
app.get("/users/:userId", (req, res) => {
  const loggedInUserId = req.params.userId;

  User.find({ _id: { $ne: loggedInUserId } })
    .then((users) => {
      res.status(200).json(users);
    })
    .catch((err) => {
      console.log("Error retrieving users", err);
      res.status(500).json({ message: "Error retrieving users" });
    });
});

//endpoint to send a request to a user
app.post("/friend-request", async (req, res) => {
  const { currentUserId, selectedUserId } = req.body;

  try {
    //update the recepient's friendRequestsArray!
    await User.findByIdAndUpdate(selectedUserId, {
      $push: { freindRequests: currentUserId },
    });

    //update the sender's sentFriendRequests array
    await User.findByIdAndUpdate(currentUserId, {
      $push: { sentFriendRequests: selectedUserId },
    });

    res.sendStatus(200);
  } catch (error) {
    res.sendStatus(500);
  }
});

//endpoint to show all the friend-requests of a particular user
app.get("/friend-request/:userId", async (req, res) => {
  try {
    const { userId } = req.params;

    //fetch the user document based on the User id
    const user = await User.findById(userId)
      .populate("freindRequests", "name email image")
      .lean();

    const freindRequests = user.freindRequests;

    res.json(freindRequests);
  } catch (error) {
    console.log(error);
    res.status(500).json({ message: "Internal Server Error" });
  }
});

//endpoint to accept a friend-request of a particular person
app.post("/friend-request/accept", async (req, res) => {
  try {
    const { senderId, recepientId } = req.body;

    //retrieve the documents of sender and the recipient
    const sender = await User.findById(senderId);
    const recepient = await User.findById(recepientId);

    sender.friends.push(recepientId);
    recepient.friends.push(senderId);

    recepient.freindRequests = recepient.freindRequests.filter(
      (request) => request.toString() !== senderId.toString()
    );

    sender.sentFriendRequests = sender.sentFriendRequests.filter(
      (request) => request.toString() !== recepientId.toString()
    );

    await sender.save();
    await recepient.save();

    res.status(200).json({ message: "Friend Request accepted successfully" });
  } catch (error) {
    console.log(error);
    res.status(500).json({ message: "Internal Server Error" });
  }
});

//endpoint to access all the friends of the logged in user!
app.get("/accepted-friends/:userId", async (req, res) => {
  try {
    const { userId } = req.params;
    const user = await User.findById(userId).populate(
      "friends",
      "name email image"
    );
    const acceptedFriends = user.friends;
    res.json(acceptedFriends);
  } catch (error) {
    console.error(error);
    res.status(500).json({ error: "Internal Server Error" });
  }
});


// check friends 
app.get("/friend-requests/sent/:userId", async (req, res) => {
  try {
    const { userId } = req.params;
    const user = await User.findById(userId).populate("sentFriendRequests", "name email image").lean();

    const sentFriendRequests = user.sentFriendRequests;

    res.json(sentFriendRequests);
  } catch (error) {
    console.log("error", error);
    res.status(500).json({ error: "Internal Server" });
  }
})

app.get("/friends/:userId", (req, res) => {
  try {
    const { userId } = req.params;

    User.findById(userId).populate("friends").then((user) => {
      if (!user) {
        return res.status(404).json({ message: "User not found" })
      }

      const friendIds = user.friends.map((friend) => friend._id);

      res.status(200).json(friendIds);
    })
  } catch (error) {
    console.log("error", error);
    res.status(500).json({ message: "internal server error" })
  }
})


// chat routes
app.post("/addmsg/", async (req, res, next) => {
  try {
    const { from, to, message } = req.body;
    const data = await Messages.create({
      message: { text: message },
      users: [from, to],
      sender: from,
    });

    if (data) return res.json({ msg: "Message added successfully." });
    else return res.json({ msg: "Failed to add message to the database" });
  } catch (ex) {
    next(ex);
  }
});


app.post("/getmsg/", async (req, res, next) => {
  try {
    const { from, to } = req.body;

    const messages = await Messages.find({
      users: {
        $all: [from, to],
      },
    }).sort({ updatedAt: 1 });

    const projectedMessages = messages.map((msg) => {
      return {
        fromSelf: msg.sender.toString() === from,
        message: msg.message.text,
      };
    });
    res.json(projectedMessages);
  } catch (ex) {
    next(ex);
  }
});


const server = app.listen(port, (e) => {
  console.log(`server is running on port ${port}`)
})

const io = socket(server, {
  cors: {
    origin: "*",
    credentials: true,
  },
});

global.onlineUsers = new Map();
io.on("connection", (socket) => {
  global.chatSocket = socket;
  socket.on("add-user", (userId) => {
    onlineUsers.set(userId, socket.id);
  });

  socket.on("send-msg", (data) => {
    const sendUserSocket = onlineUsers.get(data.to);
    if (sendUserSocket) {
      socket.to(sendUserSocket).emit("msg-recieve", data.msg);
    }
  });
});



// add image in user profile 
const fs = require('fs');
const path = require('path');

// Ensure the uploads directory exists
const uploadsDir = path.join(__dirname, './uploads');
if (!fs.existsSync(uploadsDir)) {
  fs.mkdirSync(uploadsDir);
}

// Configure multer for file uploads
const storage = multer.diskStorage({
  destination: function (req, file, cb) {
    cb(null, './src/uploads/');
  },
  filename: function (req, file, cb) {
    cb(null, Date.now() + path.extname(file.originalname)); // Append the file extension
  }
});

const upload = multer({ storage: storage });

// Serve static files from the uploads directory
app.use('/uploads', express.static(path.join(__dirname, 'uploads')));


// POST endpoint to upload an image
app.post('/upload-image/:userId', upload.single('image'), async (req, res) => {
  const userId = req.params.userId;

  try {
    const user = await User.findById(userId);

    if (!user) {
      return res.status(404).json({ message: 'User not found' });
    }

    user.image = `/uploads/${req.file.filename}`; // Store the file path in the user document

    await user.save();

    res.status(200).json({ message: 'Image uploaded successfully', imageUrl: user.image });
  } catch (error) {
    console.error(error);
    res.status(500).json({ message: 'Server error' });
  }
});



// posts 
// Ensure the posts directory exists within uploads
const postsDir = path.join(__dirname, './uploads/posts');
if (!fs.existsSync(postsDir)) {
  fs.mkdirSync(postsDir, { recursive: true });
}

// Configure multer for file uploads
const postsStorage = multer.diskStorage({
  destination: function (req, file, cb) {
    cb(null, postsDir);
  },
  filename: function (req, file, cb) {
    cb(null, Date.now() + path.extname(file.originalname)); // Append the file extension
  }
});

const uploadPosts = multer({ storage: postsStorage });

// Serve static files from the uploads directory
// app.use('/uploads', express.static(path.join(__dirname, 'uploads')));


// Create posts endpoint
app.post('/posts', uploadPosts.single('image'), async (req, res) => {
  try {
    const { description, user } = req.body;
    const imageUrl = req.file ? `/uploads/posts/${req.file.filename}` : null;

    const newPost = new Post({
      description,
      imageUrl,
      user,
      createdAt: new Date()
    });

    await newPost.save();

    res.status(201).json(newPost);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});


// Get posts by user ID sorted by creation date (newest first)
app.get('/posts/user/:userId', async (req, res) => {
  try {
    const { userId } = req.params;
    const posts = await Post.find({ user: userId }).sort({ createdAt: -1 }).populate('user');
    res.status(200).json(posts);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// Get all posts sorted by newest first, excluding posts by the provided user ID, and populate user information
app.get('/posts', async (req, res) => {
  const { userId } = req.query;
  try {
    const query = userId ? { user: { $ne: userId } } : {};
    const posts = await Post.find(query).sort({ createdAt: -1 }).populate('user');
    res.status(200).json(posts);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});


// Get all posts sorted by newest first and populate user information
// app.get('/posts', async (req, res) => {
//   try {
//     const posts = await Post.find().sort({ createdAt: -1 }).populate('user');
//     res.status(200).json(posts);
//   } catch (error) {
//     res.status(500).json({ error: error.message });
//   }
// });


// Update user profession and address
app.post('/updateUser', async (req, res) => {
  const { userId, profession, address } = req.body;

  // Validate input
  if (!userId || !profession || !address) {
      return res.status(400).json({ message: 'User ID, profession, and address are required' });
  }

  try {
      // Find user by ID and update
      const user = await User.findByIdAndUpdate(userId, { profession, address }, { new: true });

      if (!user) {
          return res.status(404).json({ message: 'User not found' });
      }

      res.status(200).json({ message: 'User updated successfully', user });
  } catch (error) {
      console.error('Error updating user:', error);
      res.status(500).json({ message: 'Internal server error' });
  }
});