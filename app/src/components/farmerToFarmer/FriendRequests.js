import { ActivityIndicator, SafeAreaView, ScrollView, StyleSheet, Text, TextInput, ToastAndroid, TouchableOpacity, View, RefreshControl } from 'react-native';
import React, { useEffect, useState, useCallback } from 'react';
import { useNavigation } from '@react-navigation/native';
import Ionicons from 'react-native-vector-icons/Ionicons';
import Feather from 'react-native-vector-icons/Feather';
import AsyncStorage from '@react-native-async-storage/async-storage';
import Config from 'react-native-config';
import axios from 'axios';
import FarmerConnect from './components/FarmerConnect';
import FriendRequestComp from './components/FriendRequestComp';
import BottomNavigation from '../../navigation/BottomNavigation';

const FriendRequests = () => {
    const [userId, setUserId] = useState('');
    const navigation = useNavigation();
    const [isLoading, setIsLoading] = useState(true);
    const [searchText, setSearchText] = useState('');
    const [friendRequests, setFriendRequests] = useState([]);
    const [accepted, setAccepted] = useState(false);
    const [refreshing, setRefreshing] = useState(false);

    // show toasted message
    const showToast = (message) => {
        ToastAndroid.showWithGravityAndOffset(
            message,
            ToastAndroid.LONG,
            ToastAndroid.BOTTOM,
            25,
            50
        );
    };

    const fetchFriendRequests = async () => {
        try {
            const storedUserData = await AsyncStorage.getItem('userData');
            const userData = JSON.parse(storedUserData);
            setUserId(userData._id);

            const response = await axios.get(
                `${Config.API_URL}/friend-request/${userData._id}`
            );
            if (response.status === 200) {
                const friendRequestsData = response.data.map((friendRequest) => ({
                    _id: friendRequest._id,
                    name: friendRequest.name,
                    email: friendRequest.email,
                    image: friendRequest.image,
                }));

                setFriendRequests(friendRequestsData);
            }
        } catch (err) {
            console.log("error message", err);
            showToast(`Network error!`);
        } finally {
            setIsLoading(false);
        }
    };

    useEffect(() => {
        fetchFriendRequests();
    }, [accepted]);

    const onRefresh = useCallback(() => {
        setRefreshing(true);
        fetchFriendRequests().then(() => setRefreshing(false));
    }, []);

    // accept friend requests 
    const acceptRequest = async (friendRequestId) => {
        try {
            const response = await fetch(
                `${Config.API_URL}/friend-request/accept`,
                {
                    method: "POST",
                    headers: {
                        "Content-Type": "application/json",
                    },
                    body: JSON.stringify({
                        senderId: friendRequestId,
                        recepientId: userId,
                    }),
                }
            );

            if (response.ok) {
                setAccepted(!accepted);
                showToast(`Request accepted!`);
            }
        } catch (err) {
            console.log("error accepting the friend request", err);
            showToast(`Network error!`);
        }
    };

    // search functionality
    const clearSearch = () => {
        setSearchText('');
    };

    const handleSearch = () => {
        setSearchText('');
    };

    return (
        <SafeAreaView style={{ height: '100%', backgroundColor: '#ffffff' }}>
            {isLoading ? (
                <ActivityIndicator style={styles.loader} size="large" color="#46c67c" />
            ) : (
                <ScrollView
                    refreshControl={
                        <RefreshControl refreshing={refreshing} onRefresh={onRefresh} />
                    }
                >
                    <View style={styles.mainContainer}>
                        <View style={styles.topStatusBar}>
                            <View style={styles.topLeft}>
                                <TouchableOpacity style={styles.backButton} onPress={() => navigation.goBack()}>
                                    <Ionicons name="arrow-back" size={22} color="#585C60" />
                                </TouchableOpacity>
                                <Text style={styles.heading}>Connection Requests</Text>
                            </View>
                        </View>

                        <View style={styles.searchContainer}>
                            <View style={styles.searchBarLeft}>
                                <TextInput
                                    style={styles.searchBar}
                                    placeholder="Search..."
                                    placeholderTextColor={'#605f63'}
                                    value={searchText}
                                    onChangeText={(text) => setSearchText(text)}
                                />
                                {searchText ? (
                                    <TouchableOpacity onPress={clearSearch} style={styles.cutButton}>
                                        <Ionicons name="close" size={20} color="#605f63" />
                                    </TouchableOpacity>
                                ) : null}
                            </View>
                            <TouchableOpacity onPress={handleSearch} style={styles.searchButton}>
                                <Ionicons name="search" size={20} color="#605f63" />
                            </TouchableOpacity>
                        </View>

                        <View style={styles.farmerProfilesContainer}>
                            <View style={styles.docContainer}>
                                {friendRequests.length > 0 ? (
                                    friendRequests.map((item, index) => (
                                        <FriendRequestComp key={index} item={item} id={userId} acceptRequest={acceptRequest} />
                                    ))
                                ) : (
                                    <Text style={{ color: '#000000', textAlign:'center' }}>No any Connection request received</Text>
                                )}
                            </View>
                        </View>
                    </View>
                </ScrollView>
            )}
            <BottomNavigation/>
        </SafeAreaView>
    );
};

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
        justifyContent: 'space-between',
        flexDirection: 'row',
        gap: 5,
        paddingHorizontal: 10,
        paddingVertical: 10,
        backgroundColor: '#ffffff'
    },
    backButton: {
        paddingVertical: 10,
        paddingHorizontal: 10
    },
    heading: {
        color: '#000000',
        fontSize: 16,
        fontWeight: '600'
    },
    topLeft: {
        flexDirection: 'row',
        alignItems: 'center'
    },
    friendRequests: {
        marginRight: 12,
        padding: 4
    },
    searchContainer: {
        marginHorizontal: 20,
        marginBottom: 20,
        marginTop: 5,
        flexDirection: 'row',
        alignItems: 'center',
        borderRadius: 5,
        backgroundColor: '#f8f7fd',
        height: 40
    },
    searchBarLeft: {
        flexDirection: 'row',
        alignItems: 'center',
        width: '86%',
    },
    searchBar: {
        display: 'flex',
        paddingVertical: 5,
        color: '#605f63',
        fontSize: 14,
        paddingHorizontal: 10,
        width: '90%'
    },
    cutButton: {
        height: '100%',
    },
    searchButton: {
        width: '12%',
        height: '100%',
        display: 'flex',
        justifyContent: 'center',
        alignItems: 'center',
    },
    farmerProfilesContainer: {
        marginHorizontal: 20,
        alignItems: 'center',
        gap: 12
    },
    docContainer: {
        width: '100%',
    }
});

export default FriendRequests;
