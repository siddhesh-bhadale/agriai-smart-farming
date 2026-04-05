import React, { useState, useEffect } from 'react';
import { View, TextInput, Button, Image, Text, StyleSheet, Alert, SafeAreaView, ScrollView, ActivityIndicator, TouchableOpacity, ToastAndroid } from 'react-native';
import axios from 'axios';
import { launchImageLibrary } from 'react-native-image-picker';
import Config from 'react-native-config';
import AsyncStorage from '@react-native-async-storage/async-storage';
import Ionicons from 'react-native-vector-icons/Ionicons'
import BottomNavigation from '../../navigation/BottomNavigation';

const CreatePostForm = () => {
    const [description, setDescription] = useState('');
    const [user, setUser] = useState(''); // Assuming user ID is entered manually
    const [imageUri, setImageUri] = useState(null);
    const [loading, setLoading] = useState(false);
    const [userData, setUserData] = useState(null);


    // tosted message 
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
                setLoading(false)
            } catch (error) {
                console.error('Error loading userData:', error);
                setLoading(false)
            }
        };
        loadUserData();
    }, []);



    const selectImage = () => {
        launchImageLibrary({ mediaType: 'photo' }, (response) => {
            if (response.didCancel) {
                console.log('User cancelled image picker');
            } else if (response.error) {
                console.log('ImagePicker Error: ', response.error);
            } else {
                setImageUri(response.assets[0].uri);
            }
        });
    };

    const submitForm = async () => {
        setLoading(true)
        if (!description || !imageUri || !userData) {
            showToast('All fields are required');
            setLoading(false)
            return;
        }

        const formData = new FormData();
        formData.append('description', description);
        formData.append('user', userData._id);
        formData.append('image', {
            uri: imageUri,
            name: `photo_${Date.now()}.jpg`,
            type: 'image/jpeg',
        });

        try {
            const response = await axios.post(`${Config.API_URL}/posts`, formData, {
                headers: {
                    'Content-Type': 'multipart/form-data',
                },
            });
            // console.log('Response:', response.data);
            showToast('Post Uploaded successfully');
            setLoading(false)
        } catch (error) {
            console.error('Error:', error);
            showToast('Network Error');
            setLoading(false)
        }
    };

    return (
        <SafeAreaView style={{ backgroundColor: '#ffffff', height: '100%' }}>
            <ScrollView>
                <View style={styles.mainContainer}>
                    <View style={styles.topStatusBar}>
                        <TouchableOpacity style={styles.backButton} onPress={() => navigation.navigate('FarmerToFarmer')}>
                            <Ionicons name="arrow-back" size={22} color="#585C60" />
                        </TouchableOpacity>
                        <Text style={styles.heading}>Create Post</Text>
                    </View>

                    {/* Post upload form */}
                    <View style={styles.Form}>
                        <TouchableOpacity style={styles.uploadImage} onPress={selectImage}>
                            <Ionicons name="cloud-upload-outline" size={30} color="#46c67c" />
                            <Text style={styles.uploadImageText}>Click to Upload <Text style={{ color: '#3f484b' }}>or drag and drop</Text></Text>
                            <Text style={styles.uploadImageSubText}>(Max File size : 5 MB)</Text>
                        </TouchableOpacity>
                        <View style={styles.imageContainer}>
                            {imageUri && <Image source={{ uri: imageUri }} style={styles.image} />}
                        </View>
                        <TextInput
                            style={styles.input}
                            multiline={true}
                            numberOfLines={4}
                            placeholder="Description"
                            placeholderTextColor={'#828282'}
                            value={description}
                            onChangeText={setDescription}
                        />
                        {loading ? (
                            <ActivityIndicator size="large" color="#46c67c" />
                        ) : (
                            <TouchableOpacity style={styles.button} title="Create Post" onPress={submitForm} >
                                <Text style={styles.buttonText}>Create Post</Text>
                            </TouchableOpacity>
                        )}
                    </View>
                </View>
            </ScrollView>
            <BottomNavigation/>
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
        // borderWidth:1,
        paddingVertical: 10,
        paddingHorizontal: 10
    },
    heading: {
        color: '#000000',
        fontSize: 16,
        fontWeight: '600'
    },

    //form 
    Form: {
        flex: 1,
        padding: 20,
        justifyContent: 'center',
    },
    uploadImage: {
        // borderWidth: 1,
        height: 160,
        alignItems: 'center',
        justifyContent: 'center',
        backgroundColor: '#e9f1f1',
        borderRadius: 8
    },
    uploadImageText: {
        color: '#46c67c'
    },
    uploadImageSubText: {
        color: '#828282',
        fontSize: 13
    },
    imageContainer: {
        // borderWidth: 1,
        alignItems: 'center',
        justifyContent: 'center',
        marginVertical: 10
    },
    image: {
        width: 150,
        height: 150,
        marginVertical: 10,
    },
    input: {
        // height: 40,
        // borderColor: 'gray',
        // borderWidth: 1,
        marginBottom: 10,
        paddingHorizontal: 15,
        color: '#000000',
        borderRadius: 7,
        textAlignVertical: 'top',
        backgroundColor:'#e9f1f1'
    },

    //add post button
    button: {
        marginTop: 10,
        backgroundColor: '#46c67c',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        paddingVertical: 10,
        borderRadius: 7
    },
    buttonText: {
        fontSize: 15,
        fontWeight: '500',
        color: '#ffffff',
    },
});

export default CreatePostForm;
