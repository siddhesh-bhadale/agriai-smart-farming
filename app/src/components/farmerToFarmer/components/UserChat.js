import { Image, RootTagContext, StyleSheet, Text, TouchableOpacity, View } from 'react-native'
import React from 'react'
import { useNavigation } from '@react-navigation/native'
import Config from 'react-native-config';

const UserChat = ({ item, userId, socket }) => {
    const navigation = useNavigation();
    return (
        <TouchableOpacity style={styles.chatContainer} onPress={() => { navigation.navigate('Chat', { item: item, userId: userId, socket: socket }) }}>
            <View style={styles.progileContainer}>
                <Image style={styles.profileImage} source={
                    item.image
                        ? { uri: `${Config.API_URL}${item.image}` }
                        : require('../../../assets/icon/user.jpg')
                } />
            </View>
            <View style={styles.userInfo}>
                <View style={styles.Divider}>
                    <Text style={styles.userName}>{item.name}</Text>
                    <Text style={styles.time}>7:39 am</Text>
                </View>
                <View style={styles.Divider}>
                    <Text style={styles.userInfoText}>Farmer</Text>
                    <Text style={styles.messageCount}>2</Text>
                </View>
            </View>
        </TouchableOpacity>
    )
}

const styles = StyleSheet.create({
    chatContainer: {
        borderBottomWidth: 0.2,
        borderBottomColor: '#e0e0e0',
        width: '100%',
        alignItems: 'center',
        justifyContent: 'flex-start',
        flexDirection: 'row',
        paddingVertical: 10
    },
    progileContainer: {
        width: 47,
        height: 47,
        // borderWidth: 2,
        // borderRadius: 100,
        // borderColor: '#eeeeee'
    },
    profileImage: {
        width: '100%',
        height: '100%',
        borderRadius: 100,
        borderWidth: 2.5,
        borderColor: '#e9f1f1'
    },
    Divider: {
        flexDirection: 'row',
        justifyContent: 'space-between',
        width: '92%',
        alignContent: 'center'
    },
    userName: {
        fontSize: 15,
        fontWeight: '500',
        color: '#3f484b'
    },
    time: {
        color: '#828282',
        fontSize: 12
    },
    userInfo: {
        marginLeft: 10,
        justifyContent: 'center',
        // borderWidth:1,
    },
    userInfoText: {
        color: '#828282',
        fontSize: 12,
        fontWeight: '400',
        paddingTop: 2
    },
    messageCount: {
        backgroundColor: '#46c67c',
        borderRadius: 100,
        paddingHorizontal: 6,
        paddingVertical: 2,
        fontSize: 10,
        fontWeight: '700'
    }
})

export default UserChat

