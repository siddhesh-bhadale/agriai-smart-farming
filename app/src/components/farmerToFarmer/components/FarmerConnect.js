import { Image, StyleSheet, Text, ToastAndroid, TouchableOpacity, View } from 'react-native'
import React, { useState, useEffect } from 'react'
import Config from 'react-native-config';
import axios from 'axios';

const FarmerConnect = ({ item, id, sendFriendRequest, acceptRequest }) => {
    const [requestSent, setRequestSent] = useState(false);


    //load user is friend or not 
    const [friendRequests, setFriendRequests] = useState([]);
    const [userFriends, setUserFriends] = useState([]);
    const [recivedRequests, setRecivedRequests] = useState([]);

    useEffect(() => {
        const fetchFriendRequests = async () => {
            try {
                const response = await fetch(
                    `${Config.API_URL}/friend-requests/sent/${id}`
                );

                const data = await response.json();
                if (response.ok) {
                    setFriendRequests(data);
                } else {
                    console.log("error", response.status);
                }
            } catch (error) {
                console.log("error", error);
            }
        };

        fetchFriendRequests();
    }, []);

    useEffect(() => {
        const fetchUserFriends = async () => {
            try {
                const response = await fetch(`${Config.API_URL}/friends/${id}`);

                const data = await response.json();

                if (response.ok) {
                    setUserFriends(data);
                } else {
                    console.log("error retrieving user friends", response.status);
                }
            } catch (error) {
                console.log("Error message", error);
            }
        };

        fetchUserFriends();
    }, []);


    // load recived friend request
    useEffect(() => {
        const fetchFriendRequests = async () => {
            try {
                const response = await axios.get(
                    `${Config.API_URL}/friend-request/${id}`
                );
                if (response.status === 200) {
                    const friendRequestsData = response.data.map((friendRequest) => ({
                        _id: friendRequest._id,
                        name: friendRequest.name,
                        email: friendRequest.email,
                        image: friendRequest.image,
                    }));

                    setRecivedRequests(friendRequestsData);
                }
            } catch (err) {
                console.log("error message", err);
            }
        };

        fetchFriendRequests();
    }, []);


    return (
        <View style={styles.container}>
            <View style={styles.left}>
                <View style={styles.progileContainer}>
                    <Image style={styles.profileImage} source={
                        item.image
                            ? { uri: `${Config.API_URL}${item.image}` }
                            : require('../../../assets/icon/user.jpg')
                    } />
                </View>
                <View style={styles.userInfo}>
                    <Text style={styles.userName}>{item.name}</Text>
                    <Text style={styles.userInfoText}>Farmer</Text>
                </View>
            </View>
            {/* <TouchableOpacity style={styles.button} onPress={() => sendFriendRequest(id, item._id, item.name)}>
                <Text style={styles.buttonText}>Connect</Text>
            </TouchableOpacity> */}


            {userFriends.includes(item._id) ? (
                <TouchableOpacity style={[styles.button, { backgroundColor: "#82CD47", }]} onPress={() => { }}>
                    <Text style={styles.buttonText}>Friends</Text>
                </TouchableOpacity>
            ) : requestSent || friendRequests.some((friend) => friend._id === item._id) ? (
                <TouchableOpacity style={styles.button} onPress={() => { }}>
                    <Text style={styles.buttonText}>Sended</Text>
                </TouchableOpacity>
            ) : recivedRequests.some((friend) => friend._id === item._id) ? (
                <TouchableOpacity style={styles.button} onPress={() => {acceptRequest(item._id) }}>
                    <Text style={styles.buttonText}>Accept</Text>
                </TouchableOpacity>
            ) : (
                <TouchableOpacity style={styles.button} onPress={() => sendFriendRequest(id, item._id, item.name)}>
                    <Text style={styles.buttonText}>Connect</Text>
                </TouchableOpacity>
            )}
        </View>
    )
}

const styles = StyleSheet.create({
    container: {
        alignItems: 'center',
        justifyContent: 'space-between',
        flexDirection: 'row',
        width: '100%'
    },
    left: {
        flexDirection: 'row',

    },
    progileContainer: {
        width: 55,
        height: 55
    },
    profileImage: {
        width: '100%',
        height: '100%',
        borderRadius:100,
        borderWidth:2.5,
        borderColor:'#e9f1f1'
    },
    userName: {
        fontSize: 15,
        fontWeight: '600',
        color: '#3f484b'
    },
    userInfo: {
        marginLeft: 10,
        justifyContent: 'center',
        // borderWidth:1,
        maxWidth: 150
    },
    userInfoText: {
        color: '#000000',
        fontSize: 12
    },
    button: {
        backgroundColor: '#46c67c',
        paddingVertical: 9,
        paddingHorizontal: 34,
        borderRadius: 7,
        width:120,
        alignItems:'center'
    },
    buttonText: {
        color: '#ffffff',
        fontSize: 14,
        fontWeight: '500'
    }
})

export default FarmerConnect
