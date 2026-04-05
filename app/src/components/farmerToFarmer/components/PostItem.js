import { StyleSheet, Text, Image, View, TouchableOpacity } from 'react-native'
import React, { useState, useEffect } from 'react'
import MaterialCommunityIcons from 'react-native-vector-icons/MaterialCommunityIcons'
import Feather from 'react-native-vector-icons/Feather'
import EvilIcons from 'react-native-vector-icons/EvilIcons'
import Fontisto from 'react-native-vector-icons/Fontisto'
import Ionicons from 'react-native-vector-icons/Ionicons'
import AntDesign from 'react-native-vector-icons/AntDesign'
import Config from 'react-native-config'


const PostItem = ({ item }) => {
    // console.log(item)

    const convertToTime = (isoString) => {
        const date = new Date(isoString);
        return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', hour12: true });
      };

    return (
        <View>
            <View style={styles.postItem}>
                <View style={styles.userData}>
                    <View style={styles.postUserProfileContainer}>

                        {/* <Image style={styles.postUserProfile} source={require('../../../assets/icon/user.jpg')} /> */}
                        <Image style={styles.postUserProfile} source={
                            item.user.image
                                ? { uri: `${Config.API_URL}${item.user.image}` }
                                : require('../../../assets/icon/user.jpg')
                        } />
                    </View>
                    <View style={styles.userDetails}>
                        <Text style={styles.userName}>{item.user.name}</Text>
                        <Text style={styles.userProfation}>{item.user.profession ? item.user.profession : '--'}</Text>
                        <Text style={styles.postTime}>{convertToTime(item.createdAt)}</Text>
                    </View>
                </View>
                <View style={styles.postData}>
                    <Text style={styles.postText}>
                        {item.description}
                    </Text>
                    <View style={styles.postImageContainer}>
                        <Image style={styles.postImage} source={{ uri: `${Config.API_URL}${item.imageUrl}` }} />
                    </View>
                </View>
                <View style={styles.buttonContainer}>
                    <TouchableOpacity style={styles.icon}>
                        <EvilIcons name="like" size={27} color="#3f484b" />
                        <Text style={styles.buttonLable}>Like</Text>
                    </TouchableOpacity>
                    <TouchableOpacity style={styles.icon}>
                        <Fontisto name="commenting" size={17} color="#3f484b" />
                        <Text style={styles.buttonLable}>Comment</Text>
                    </TouchableOpacity>
                    <TouchableOpacity style={styles.icon}>
                        <Ionicons name="arrow-redo-outline" size={22} color="#3f484b" />
                        <Text style={styles.buttonLable}>Share</Text>
                    </TouchableOpacity>
                    <TouchableOpacity style={styles.icon}>
                        <Feather name="send" size={20} color="#3f484b" />
                        <Text style={styles.buttonLable}>Send</Text>
                    </TouchableOpacity>
                </View>
            </View>
        </View>
    )
}

export default PostItem

const styles = StyleSheet.create({
    postItem: {
        width: '100%',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        marginBottom: 20,
        paddingHorizontal: 15,
        paddingBottom: 10,
        paddingTop: 15,
        borderRadius: 7,
        backgroundColor: '#e9f1f1'
    },
    userData: {
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'flex-start',
        flexDirection: 'row',
        width: '100%',
        gap: 13
    },
    postUserProfileContainer: {
        height: 50,
        width: 50,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        overflow: 'hidden',
        borderRadius: 100
    },
    postUserProfile: {
        width: '100%',
        height: '100%',
        borderWidth: 2,
        borderColor: '#ffffff',
        borderRadius: 100
    },
    userName: {
        color: '#000000',
        fontSize: 15,
        fontWeight: '500',
    },
    userProfation: {
        color: '#585C60',
        fontSize: 13,
        fontWeight: '400',
    },
    postTime: {
        color: '#585C60',
        fontSize: 13,
        fontWeight: '300'
    },
    postData: {
        width: '100%',
        marginTop: 14
    },
    postText: {
        color: '#000000',
        fontSize: 14,
        fontWeight: '400',
        letterSpacing: 0.5,
        lineHeight: 18,
        marginBottom: 15
    },
    postImageContainer: {
        height: 260,
        width: '100%',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        overflow: 'hidden',
    },
    postImage: {
        width: '100%',
        height: '100%'
    },

    //post item buttons
    buttonContainer: {
        display: 'flex',
        alignItems: 'flex-end',
        justifyContent: 'space-between',
        flexDirection: 'row',
        width: '100%',
        marginVertical: 10
    },
    icon: {
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        // borderWidth:1,
        // paddingHorizontal:10
    },
    buttonLable: {
        color: '#3f484b',
        fontSize: 11
    },
})