import { Image, SafeAreaView, ScrollView, StyleSheet, Text, TouchableOpacity, View, ActivityIndicator, ToastAndroid, RefreshControl, FlatList } from 'react-native';
import React, { useState, useEffect } from 'react';
import { useNavigation } from '@react-navigation/native';
import Ionicons from 'react-native-vector-icons/Ionicons';
import AsyncStorage from '@react-native-async-storage/async-storage';
import MaterialIcons from 'react-native-vector-icons/MaterialIcons';
import { launchImageLibrary } from 'react-native-image-picker';
import axios from 'axios';
import Config from 'react-native-config';
import PostItem from './components/PostItem';
import BottomNavigation from '../../navigation/BottomNavigation';

const Profile = () => {
    const navigation = useNavigation();
    const [userData, setUserData] = useState(null);
    const [isLoading, setIsLoading] = useState(true);
    const [imageUri, setImageUri] = useState(null);
    const [uploadedImageUrl, setUploadedImageUrl] = useState(null);
    const [posts, setPosts] = useState([]);
    const [refreshing, setRefreshing] = useState(false);

    // Toast message 
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
                setIsLoading(false);
            } catch (error) {
                console.error('Error loading userData:', error);
                setIsLoading(false);
            }
        };

        loadUserData();
    }, []);

    const fetchPosts = async () => {
        try {
            if (userData) {
                const response = await axios.get(`${Config.API_URL}/posts/user/${userData._id}`);
                setPosts(response.data);
                setIsLoading(false);
            }
        } catch (error) {
            console.error(error);
            showToast('Failed to load posts');
            setIsLoading(false);
        }
    };

    useEffect(() => {
        fetchPosts();
    }, [userData]);

    const onRefresh = async () => {
        setRefreshing(true);
        await fetchPosts();
        setRefreshing(false);
    };

    const pickImage = () => {
        launchImageLibrary({ mediaType: 'photo' }, (response) => {
            if (response.didCancel) {
                console.log('User cancelled image picker');
            } else if (response.error) {
                console.log('ImagePicker Error: ', response.error);
            } else {
                const source = response.assets[0].uri;
                setImageUri(source);
                uploadImage(source);
            }
        });
    };

    const uploadImage = async (source) => {
        setIsLoading(true);
        if (!source) {
            showToast('No image selected');
            return;
        }

        const formData = new FormData();
        formData.append('image', {
            uri: source,
            name: 'photo.jpg',
            type: 'image/jpeg'
        });

        try {
            const response = await axios.post(`${Config.API_URL}/upload-image/${userData._id}`, formData, {
                headers: {
                    'Content-Type': 'multipart/form-data'
                }
            });

            if (response.status === 200) {
                const jsonValue = await AsyncStorage.getItem('userData');
                if (jsonValue != null) {
                    const userData = JSON.parse(jsonValue);
                    userData.image = response.data.imageUrl;
                    const updatedJsonValue = JSON.stringify(userData);
                    await AsyncStorage.setItem('userData', updatedJsonValue);
                    setUserData(JSON.parse(updatedJsonValue));
                    showToast(`Image uploaded successfully`);
                    setIsLoading(false);
                }
            }
        } catch (error) {
            console.error('Error uploading image: ', error);
            showToast('Error uploading image');
            setIsLoading(false);
        }
    };

    const renderItem = ({ item }) => (
        <PostItem item={item} />
    );

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
                            <TouchableOpacity style={styles.backButton} onPress={() => navigation.navigate('FarmerToFarmer')}>
                                <Ionicons name="arrow-back" size={22} color="#585C60" />
                            </TouchableOpacity>
                            <Text style={styles.heading}>Profile</Text>
                        </View>
                        <View style={styles.backgroundPoster}>
                            <TouchableOpacity style={styles.posterEditButton} onPress={() => {navigation.navigate('UpdateProfileForm')}}>
                                <MaterialIcons name="edit" size={17} color="#0A66C2" />
                            </TouchableOpacity>
                        </View>
                        <View style={styles.container}>
                            {uploadedImageUrl && <Image source={{ uri: uploadedImageUrl }} style={{ width: 200, height: 200 }} />}
                            <TouchableOpacity style={styles.profileImageContainer} onPress={pickImage}>
                                <Image style={styles.profileImage} source={
                                    userData.image
                                        ? { uri: `${Config.API_URL}${userData.image}` }
                                        : require('../../assets/icon/user.jpg')
                                } />
                            </TouchableOpacity>
                            <View style={styles.userInfo}>
                                <Text style={styles.userName}>{userData.name}</Text>
                                <Text style={styles.userProfation}>{userData.profession ? userData.profession : '--'}</Text>
                                {/* <Text style={styles.userExperience}>Experience - 4 year</Text> */}
                                <Text style={styles.userAddress}>{userData.address ? userData.address : '--'}</Text>
                                {/* <Text style={styles.userFollowers}>2,900 followers</Text> */}
                            </View>
                            <TouchableOpacity style={styles.addPostButton} onPress={() => { navigation.navigate('PostForm') }}>
                                <Text style={styles.addPostButtonText}>Add Post</Text>
                            </TouchableOpacity>
                            <View style={styles.myPostContainer}>
                                <Text style={styles.myPostText}>Uploaded Posts :</Text>
                                <View style={styles.posts}>
                                    {posts.length > 0 ? (
                                        posts.map((item, index) => (
                                            <PostItem key={index} item={item} />
                                        ))
                                    ) : (
                                        <Text style={{ color: '#000000', textAlign: 'center' }}>No any Post uploaded!</Text>
                                    )}
                                </View>
                            </View>
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
    backgroundPoster: {
        position: 'relative',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        overflow: 'hidden',
        width: '100%',
        height: 120,
        marginTop: 10,
        backgroundColor: '#46c67c'
    },
    backgroundPosterImage: {
        height: '100%',
        width: '100%'
    },
    posterEditButton: {
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        width: 28,
        height: 28,
        borderRadius: 100,
        backgroundColor: '#ffffff',
        position: 'absolute',
        top: 15,
        right: 15
    },
    container: {
        position: 'relative',
        top: -60,
    },
    profileImageContainer: {
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        overflow: 'hidden',
        width: 130,
        height: 130,
        borderRadius: 100,
        borderWidth: 2,
        borderColor: '#e9f1f1',
        marginHorizontal: 10,
        marginBottom: 10,
        backgroundColor: '#ffffff'
    },
    profileImage: {
        width: '100%',
        height: '100%'
    },
    userInfo: {
        marginHorizontal: 15,
    },
    userName: {
        color: '#000000',
        fontSize: 22,
        fontWeight: '500',
        textTransform: 'uppercase',
        marginBottom: 7,
    },
    userProfation: {
        color: '#000000',
        fontSize: 15,
        fontWeight: '400',
        textTransform: 'capitalize'
    },
    userExperience: {
        color: '#000000',
        fontSize: 15,
        fontWeight: '400',
        textTransform: 'capitalize'
    },
    userAddress: {
        color: '#585C60',
        fontSize: 15,
        fontWeight: '300',
        textTransform: 'capitalize',
    },
    userFollowers: {
        color: '#46c67c',
        fontSize: 16,
        fontWeight: '700',
        marginVertical: 10
    },
    addPostButton: {
        marginHorizontal: 15,
        marginTop: 17,
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
    myPostContainer: {
        marginHorizontal: 15,
        marginTop: 30
    },
    myPostText: {
        color: "#3f484b",
        fontSize: 16,
        fontWeight: '600',
        letterSpacing: 0.3,
        marginBottom: 20
    }
});

export default Profile;
