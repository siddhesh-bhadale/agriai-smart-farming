import React, { useState, useEffect } from 'react';
import { View, Text, TextInput, Button, StyleSheet, Alert, ActivityIndicator, SafeAreaView, ScrollView, TouchableOpacity, ToastAndroid } from 'react-native';
import axios from 'axios';
import AsyncStorage from '@react-native-async-storage/async-storage';
import Config from 'react-native-config';
import Ionicons from 'react-native-vector-icons/Ionicons';
import BottomNavigation from '../../../navigation/BottomNavigation';

const UpdateProfileForm = () => {
    const [profession, setProfession] = useState('');
    const [address, setAddress] = useState('');
    const [loading, setLoading] = useState(false);
    const [userId, setUserId] = useState(null);

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

    useEffect(() => {
        const loadUserId = async () => {
            try {
                const storedUserData = await AsyncStorage.getItem('userData');
                const parsedUserData = JSON.parse(storedUserData);
                setUserId(parsedUserData._id);
            } catch (error) {
                console.error('Error loading userId:', error);
            }
        };

        loadUserId();
    }, []);

    const handleSubmit = async () => {
        if (!profession || !address) {
            showToast('Both profession and address are required.');
            return;
        }

        setLoading(true);

        try {
            const response = await axios.post(`${Config.API_URL}/updateUser`, {
                userId,
                profession,
                address
            });

            if (response.status === 200) {
                showToast('Profile updated successfully');

                const updatedJsonValue = JSON.stringify(response.data.user);
                await AsyncStorage.setItem('userData', updatedJsonValue);
                // setUserData(JSON.parse(updatedJsonValue));
                console.log(response.data.user)
            } else {
                showToast('Failed to update profile');
            }
        } catch (error) {
            console.error('Error updating profile:', error);
            showToast('An error occurred while updating profile');
        } finally {
            setLoading(false);
        }
    };

    return (
        <SafeAreaView style={{ height: '100%', backgroundColor: '#ffffff' }}>
            <ScrollView>
                <View style={styles.mainContainer}>
                    <View style={styles.topStatusBar}>
                        <TouchableOpacity style={styles.backButton} onPress={() => navigation.navigate('FarmerToFarmer')}>
                            <Ionicons name="arrow-back" size={22} color="#585C60" />
                        </TouchableOpacity>
                        <Text style={styles.heading}>Update Profile</Text>
                    </View>
                    <View style={styles.container}>
                        <Text style={styles.label}>Profession</Text>
                        <TextInput
                            style={styles.input}
                            value={profession}
                            onChangeText={setProfession}
                            placeholder="Enter your profession"
                            placeholderTextColor={'#828282'}
                        />
                        <Text style={styles.label}>Address</Text>
                        <TextInput
                            style={styles.input}
                            value={address}
                            onChangeText={setAddress}
                            placeholder="Enter your address"
                            placeholderTextColor={'#828282'}
                        />
                        {loading ? (
                            <ActivityIndicator size="large" color="#46c67c" />
                        ) : (
                            // <Button title="Update Profile" onPress={handleSubmit} />
                            <TouchableOpacity style={styles.addPostButton} onPress={handleSubmit}>
                                <Text style={styles.addPostButtonText}>Update Profile</Text>
                            </TouchableOpacity>
                        )}
                    </View>
                </View>
            </ScrollView>
            <BottomNavigation />
        </SafeAreaView>
    );
};

const styles = StyleSheet.create({
    mainContainer: {
        backgroundColor: '#ffffff',
        height: '100%',
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
        paddingTop: 10
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
    container: {
        flex: 1,
        padding: 20,
        backgroundColor: '#fff'
    },
    label: {
        fontSize: 14,
        marginBottom: 5,
        color: '#000000'
    },
    input: {
        height: 40,
        borderColor: '#ccc',
        borderWidth: 1,
        borderRadius: 7,
        marginBottom: 20,
        paddingHorizontal: 10,
        color: '#3f484b'
    },
    addPostButton: {
        // marginHorizontal: 15,
        marginTop: 10,
        backgroundColor: '#46c67c',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        paddingVertical: 10,
        borderRadius: 7
    },
    addPostButtonText: {
        fontSize: 15,
        fontWeight: '500',
        color: '#ffffff',
    },
});

export default UpdateProfileForm;
