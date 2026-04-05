import { ActivityIndicator, SafeAreaView, ScrollView, StyleSheet, Text, TextInput, ToastAndroid, TouchableOpacity, View, RefreshControl } from 'react-native';
import React, { useEffect, useState, useCallback } from 'react';
import { useNavigation } from '@react-navigation/native';
import Ionicons from 'react-native-vector-icons/Ionicons';
import Feather from 'react-native-vector-icons/Feather';
import AsyncStorage from '@react-native-async-storage/async-storage';
import Config from 'react-native-config';
import axios from 'axios';
import FarmerConnect from './components/FarmerConnect';
import BottomNavigation from '../../navigation/BottomNavigation';

const AddFarmer = () => {
    const [userId, setUserId] = useState('');
    const navigation = useNavigation();
    const [isLoading, setIsLoading] = useState(true);
    const [searchText, setSearchText] = useState('');
    const [users, setUsers] = useState([]);
    const [requestSent, setRequestSent] = useState(false);
    const [accepted, setAccepted] = useState(false);
    const [refreshing, setRefreshing] = useState(false);
    const [friendRequests, setFriendRequests] = useState([]);

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

    const fetchUsers = async () => {
        setIsLoading(true);
        const storedUserData = await AsyncStorage.getItem('userData');
        const userData = JSON.parse(storedUserData);
        setUserId(userData._id);

        try {
            const response = await axios.get(`${Config.API_URL}/users/${userData._id}`);
            setUsers(response.data);
        } catch (error) {
            console.log("error retrieving users", error);
        } finally {
            setIsLoading(false);
        }
    };

    // fetch recived requests 
    const fetchFriendRequests = async () => {
        try {
            const storedUserData = await AsyncStorage.getItem('userData');
            const userData = JSON.parse(storedUserData);

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
        fetchUsers();
        fetchFriendRequests();
    }, [userId, requestSent, accepted]);

    const onRefresh = useCallback(() => {
        setRefreshing(true);
        fetchUsers().then(() => setRefreshing(false));
    }, []);

    // send friend request
    const sendFriendRequest = async (currentUserId, selectedUserId, name) => {
        try {
            const response = await fetch(`${Config.API_URL}/friend-request`, {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify({ currentUserId, selectedUserId }),
            });

            if (response.ok) {
                setRequestSent(!requestSent);
                showToast(`Request sent to ${name}`);
            }
        } catch (error) {
            console.log("error message", error);
            showToast(`Network error!`);
        }
    };

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

    // console.log('friendRequests', friendRequests.length)

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
                                <Text style={styles.heading}>Farmers</Text>
                            </View>
                            <TouchableOpacity style={styles.friendRequests} onPress={() => { navigation.navigate("FriendRequests") }}>
                                {friendRequests.length > 0 ? (
                                    <Text style={styles.requestCounts}>{friendRequests.length}</Text>
                                ) : null}
                                <Feather name="users" size={22} color="#000000" />
                            </TouchableOpacity>
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
                            {users.length > 0 ? (
                                users.map((item, index) => (
                                    <FarmerConnect key={index} item={item} id={userId} sendFriendRequest={sendFriendRequest} acceptRequest={acceptRequest} />
                                ))
                            ) : (
                                <Text style={{ color: '#000000', textAlign: 'center' }}>No any User found!</Text>
                            )}
                        </View>
                    </View>
                </ScrollView>
            )}
            <BottomNavigation />
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
    requestCounts: {
        backgroundColor: '#46c67c',
        fontSize: 10,
        fontWeight: '700',
        width: 14,
        height: 14,
        borderRadius: 100,
        alignItems: 'center',
        justifyContent: 'center',
        textAlign: 'center',
        position: 'absolute',
        right: -3,
        zIndex: 3
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
    }
});

export default AddFarmer;
