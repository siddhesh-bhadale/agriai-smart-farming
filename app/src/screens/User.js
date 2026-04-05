import { StyleSheet, Text, View, SafeAreaView, ScrollView, ActivityIndicator, TouchableOpacity, Image, ToastAndroid } from 'react-native'
import React, { useEffect, useState } from 'react'
import { useNavigation } from '@react-navigation/native';
import Ionicons from 'react-native-vector-icons/Ionicons';
import AsyncStorage from '@react-native-async-storage/async-storage';
import BottomNavigation from '../navigation/BottomNavigation';
import config, { Config } from 'react-native-config'
const User = () => {
    const navigation = useNavigation();
    const [userData, setUserData] = useState(null);
    const [isLoading, setIsLoading] = useState(true);

    const showToast = (message) => {
        ToastAndroid.showWithGravityAndOffset(
            message,
            ToastAndroid.LONG,
            ToastAndroid.BOTTOM,
            25,
            50
        );
    };

    useEffect(() => {
        const loadUserData = async () => {
            try {
                const storedUserData = await AsyncStorage.getItem('userData');
                setUserData(JSON.parse(storedUserData));
                setIsLoading(false)
            } catch (error) {
                console.error('Error loading userData:', error);
                setIsLoading(false)
            }
        };

        loadUserData();
    }, []);

    const handleLogout = async () => {
        await AsyncStorage.removeItem('userData');
        navigation.reset({
            index: 0,
            routes: [{ name: 'Auth' }],
        });
        // navigation.navigate('Auth');
        showToast('Logout Succesfull')
    };

    return (
        <SafeAreaView style={{ height: '100%', backgroundColor: '#ffffff' }}>
            {isLoading ? (
                <ActivityIndicator style={styles.loader} size="large" color="#46c67c" />
            ) : (
                <ScrollView>
                    <View style={styles.mainContainer}>
                        <View style={styles.topStatusBar}>
                            <TouchableOpacity style={styles.backButton} onPress={() => navigation.navigate('FarmerToFarmer')}>
                                <Ionicons name="arrow-back" size={22} color="#585C60" />
                            </TouchableOpacity>
                            <Text style={styles.heading}>Settings</Text>
                        </View>

                        <View style={styles.profileContainer}>
                            <View style={styles.imageContainer}>
                                <Image style={styles.userImage} source={
                                    userData.image
                                        ? { uri: `${Config.API_URL}${userData.image}` }
                                        : require('../assets/icon/user.jpg')
                                } />
                            </View>
                            <View style={styles.profileDetails}>
                                <View style={styles.userDetails}>
                                    <Text style={styles.hello}>Hello</Text>
                                    <Text style={styles.userName}>{userData.name}</Text>
                                </View>
                                <TouchableOpacity style={styles.editButton}>
                                    <Ionicons style={styles.editButtonIcon} name="create-sharp" size={16} color="#ffffff" />
                                </TouchableOpacity>
                            </View>
                        </View>

                        <View style={styles.buttonContainer}>
                            <TouchableOpacity style={styles.navButton} onPress={() => { navigation.navigate('Issued') }}>
                                <Text style={styles.navButtonText}>Detect Diseases</Text>
                                <Ionicons style={styles.editButtonIcon} name="chevron-forward-outline" size={16} color="#000000" />
                            </TouchableOpacity>
                            <TouchableOpacity style={styles.navButton} onPress={() => { navigation.navigate('Issued') }}>
                                <Text style={styles.navButtonText}>Track Crops</Text>
                                <Ionicons style={styles.editButtonIcon} name="chevron-forward-outline" size={16} color="#000000" />
                            </TouchableOpacity>
                            <TouchableOpacity style={styles.navButton}>
                                <Text style={styles.navButtonText}>Reccomended Crop</Text>
                                <Ionicons style={styles.editButtonIcon} name="chevron-forward-outline" size={16} color="#000000" />
                            </TouchableOpacity>
                            <TouchableOpacity style={styles.navButton} onPress={() => { navigation.navigate('ChangePassword') }}>
                                <Text style={styles.navButtonText}>Change Password</Text>
                                <Ionicons style={styles.editButtonIcon} name="chevron-forward-outline" size={16} color="#000000" />
                            </TouchableOpacity>
                        </View>

                        <View style={[styles.buttonContainer, { paddingBottom: 6 }]}>
                            <TouchableOpacity style={styles.navButton} onPress={() => { handleLogout() }}>
                                <Text style={[styles.navButtonText, { color: '#F93535' }]}>Logout</Text>
                                <Ionicons style={styles.editButtonIcon} name="chevron-forward-outline" size={16} color="#F93535" />
                            </TouchableOpacity>
                        </View>
                    </View>
                </ScrollView>
            )}
            <BottomNavigation />
        </SafeAreaView>
    )
}

const styles = StyleSheet.create({
    mainContainer: {
        backgroundColor: '#ffffff',
        height: '100%'
    },
    loader: {
        flex: 1,
        justifyContent: 'center',
        alignItems: 'center',
    },
    topStatusBar: {
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'flex-start',
        flexDirection: 'row',
        gap: 5,
        paddingHorizontal: 10,
        paddingVertical: 10,
        backgroundColor: '#ffffff'
    },
    backButton: {
        // borderWidth:1,
        paddingVertical: 10,
        paddingHorizontal: 10
    },
    heading: {
        color: '#000000',
        fontSize: 16,
        fontWeight: '600'
    },

    // user profile 
    profileContainer: {
        flex: 1,
        flexDirection: 'row',
        paddingHorizontal: 15,
        gap: 10,
        paddingVertical: 35,
        backgroundColor: '#E9F1F1'
    },
    profileDetails: {
        flex: 1,
        flexDirection: 'row',
        justifyContent: 'space-between',
        alignItems: 'center'
    },

    imageContainer: {
        height: 60,
        width: 60,
        borderRadius: 100,
        overflow: 'hidden'
    },
    userImage: {
        height: "100%",
        width: "100%",
    },
    hello: {
        color: '#3f484b',
        fontWeight: '400',
        fontSize: 14
    },
    userName: {
        color: '#3f484b',
        fontSize: 16,
        fontWeight: '500',
        textTransform: 'capitalize'
    },
    editButton: {
        borderRadius: 100,
        backgroundColor: '#46c67c'
    },
    editButtonIcon: {
        paddingVertical: 7,
        paddingHorizontal: 7
    },

    // navigation buttons
    buttonContainer: {
        backgroundColor: '#eeeeee',
        paddingTop: 7,
    },
    navButton: {
        borderBottomWidth: 1,
        borderColor: '#eeeeee',
        flex: 1,
        alignItems: 'center',
        justifyContent: 'space-between',
        flexDirection: 'row',
        paddingLeft: 20,
        paddingRight: 10,
        paddingVertical: 5,
        backgroundColor: '#ffffff'
    },
    navButtonText: {
        color: '#000000',

    }
})

export default User
