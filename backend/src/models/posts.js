const mongoose = require('mongoose');

const postSchema = new mongoose.Schema({
    description: String,
    imageUrl: String,
    user: {
        type: mongoose.Schema.Types.ObjectId,
        ref: 'User'
    },
    createdAt: {
        type: Date,
        default: Date.now
    }
});

const Post = mongoose.model("Post", postSchema);

module.exports = Post