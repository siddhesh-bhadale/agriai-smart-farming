import { Image, StyleSheet, Text, ToastAndroid, TouchableOpacity, View } from 'react-native'
import React, { useState } from 'react'
import Config from 'react-native-config';

const FriendRequestComp = ({ item, id, acceptRequest }) => {
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
                    <Text style={styles.userName}>{item.name} send you a connection request</Text>
                    {/* <Text style={styles.userInfoText}>Farmer</Text> */}
                </View>
            </View>
            <TouchableOpacity style={styles.button} onPress={() => { acceptRequest(item._id) }}>
                <Text style={styles.buttonText}>Accept</Text>
            </TouchableOpacity>
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
        fontSize: 13,
        fontWeight: '500',
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
        paddingHorizontal: 28,
        borderRadius: 7
    },
    buttonText: {
        color: '#ffffff',
        fontSize: 14,
        fontWeight: '500'
    }
})

export default FriendRequestComp
