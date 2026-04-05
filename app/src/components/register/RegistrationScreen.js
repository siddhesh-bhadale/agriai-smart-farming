import React, { useState, useEffect } from 'react';
import { View, Text, TextInput, TouchableOpacity, ToastAndroid, ActivityIndicator, Image, StyleSheet, SafeAreaView, ScrollView } from 'react-native';
import { useNavigation } from '@react-navigation/native';
import axios from 'axios';
import { validateEmail, validatePassword } from '../../utils/utils';
import { API_URL } from "@env";
import Config from 'react-native-config'

const RegistrationScreen = () => {
    const navigation = useNavigation();
    const [name, setName] = useState('');
    const [email, setEmail] = useState('');
    const [password, setPassword] = useState('');
    const [confirmPassword, setConfirmPassword] = useState('');
    const [loading, setLoading] = useState(false);
    const [isLoading, setIsLoading] = useState(true);

    useEffect(() => {
        const timeout = setTimeout(() => {
            setIsLoading(false);
        }, 500);

        return () => clearTimeout(timeout); // Clear the timeout on component unmount
    }, []);

    const showToast = (message) => {
        ToastAndroid.showWithGravityAndOffset(
            message,
            ToastAndroid.LONG,
            ToastAndroid.BOTTOM,
            25,
            50
        );
    };

    const handleRegister = async () => {
        if (!name || !validateEmail(email) || !validatePassword(password) || password !== confirmPassword) {
            showToast('Invalid Input', 'Please enter valid details.');
            return;
        }

        setLoading(true);

        try {
            const response = await axios.post(`${Config.API_URL}/register`, {
                name,
                email,
                password,
            });

            if (response && response.data) {
                const { userData } = response.data;
                showToast('Registration Succesfull!');
                setLoading(false);
                navigation.navigate('Login');
            } else {
                setLoading(false);
                showToast('Registration Failed', 'Invalid response from the server');
            }
        } catch (error) {
            setLoading(false);
            console.log(error);
            showToast('Registration Failed', error.response ? error.response.data.message : 'An error occurred');
        }
    };

    return (
        <SafeAreaView style={{height:'100%', backgroundColor:'#ffffff'}}>
            {isLoading ? (
                <ActivityIndicator style={styles.loader} size="large" color="#46c67c" />
            ) : (
                <ScrollView>
                    <View style={styles.mainContainer}>
                        <Image style={styles.backgroundImage} source={require('../../assets/login/Subtract.png')} />
                        <Image style={styles.logo} source={require('../../assets/logo/logo.png')} />
                        <View style={styles.bottom_section}>
                            <Text style={styles.heading}>Registration</Text>
                            <View style={styles.form}>
                                <View style={styles.input_section}>
                                    <Text style={styles.text}>Name</Text>
                                    <TextInput
                                        style={styles.input}
                                        placeholder='Enter your name'
                                        placeholderTextColor={"#828282"}
                                        value={name}
                                        autoCapitalize='none'
                                        autoCorrect={false}
                                        onChangeText={setName}
                                    />
                                </View>

                                <View style={styles.input_section}>
                                    <Text style={styles.text}>Email</Text>
                                    <TextInput
                                        style={styles.input}
                                        placeholder='Enter your email'
                                        placeholderTextColor={"#828282"}
                                        value={email}
                                        autoCapitalize='none'
                                        autoCorrect={false}
                                        onChangeText={setEmail}
                                    />
                                </View>

                                <View style={styles.input_section}>
                                    <Text style={styles.text}>Password</Text>
                                    <TextInput
                                        style={styles.input}
                                        placeholder='Create password'
                                        placeholderTextColor={"#828282"}
                                        value={password}
                                        autoCapitalize='none'
                                        autoCorrect={false}
                                        onChangeText={setPassword}
                                        secureTextEntry
                                    />
                                </View>

                                <View style={styles.input_section}>
                                    <Text style={styles.text}>Confirm Password</Text>
                                    <TextInput
                                        style={styles.input}
                                        placeholder='Re-enter password'
                                        placeholderTextColor={"#828282"}
                                        value={confirmPassword}
                                        autoCapitalize='none'
                                        autoCorrect={false}
                                        onChangeText={setConfirmPassword}
                                        secureTextEntry
                                    />
                                </View>


                                <View style={styles.button_wrapper}>
                                    {loading ? (
                                        <ActivityIndicator size="large" color="#46c67c" />
                                    ) : (
                                        <TouchableOpacity style={styles.button} onPress={handleRegister} >
                                            <Text style={styles.button_text}>Register</Text>
                                        </TouchableOpacity>
                                    )}

                                    <Text style={styles.other_text}>Or continue with</Text>
                                    <View style={styles.other_button}>
                                        <TouchableOpacity style={styles.icon_button} >
                                            <Text style={styles.icon_button_text}><Image style={styles.icon} source={require('../../assets/icon/google.png')} /> Google</Text>
                                        </TouchableOpacity>
                                        <TouchableOpacity style={styles.icon_button}>
                                            <Text style={styles.icon_button_text}><Image style={styles.icon} source={require('../../assets/icon/facebook.png')} /> Facebook</Text>
                                        </TouchableOpacity>
                                    </View>

                                    <View style={styles.change_page}>
                                        <Text style={styles.text}>Already have an account? </Text>
                                        <TouchableOpacity style={styles.button2} onPress={() => navigation.navigate('Login')} >
                                            <Text style={styles.button_text2}>Sign in</Text>
                                        </TouchableOpacity>
                                    </View>
                                </View>
                            </View>
                        </View>
                    </View>
                </ScrollView>
            )}
        </SafeAreaView>
    );
};

const styles = StyleSheet.create({
    loader: {
        flex: 1,
        justifyContent: 'center',
        alignItems: 'center',
    },
    mainContainer: {
        backgroundColor: '#ffffff',
        width: '100%',
        height: '100%',
        position: 'relative'
    },
    backgroundImage: {
        position: 'relative',
        top: -25,
        zIndex: 2,
        resizeMode: 'cover',
        width: '100%',
    },
    logo: {
        position: 'absolute',
        top: 30,
        zIndex: 3,
        alignSelf: 'center',
        resizeMode: 'cover',
        height: 120,
        width: 130
    },
    heading: {
        color: '#1E293B',
        fontSize: 32,
        fontWeight: '800',
        textAlign: 'center',
        marginBottom: 60,
    },
    bottom_section: {
        backgroundColor: '#ffffff',
        // borderWidth: 2,
        position: 'relative',
        top: -90,
        paddingHorizontal: 31
    },
    input_section: {
        display: 'flex',
        alignItems: 'flex-start',
        justifyContent: 'center',
        width: '100',
    },
    text: {
        color: '#000000',
        fontSize: 14,
        fontWeight: '400',
        marginBottom: 1,
    },
    input: {
        borderBottomWidth: 1,
        borderBottomColor: '#000000',
        paddingBottom: 7,
        paddingHorizontal: 0,
        marginBottom: 22,
        marginTop: -7,
        color: '#000000',
        width: '100%',
        fontSize: 14,
        fontWeight: '400'
    },
    button_wrapper: {
        marginTop: 5
    },
    button: {
        borderRadius: 7,
        marginTop: -3,
        backgroundColor: '#46c67c',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        paddingVertical: 10,
        marginBottom: 10
    },
    button_text: {
        color: '#ffffff',
        fontSize: 14,
        fontWeight: '600',
        textTransform: 'uppercase'
    },
    change_page: {
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        flexDirection: 'row'
    },
    text2: {
        color: '#828282',
    },
    button_text2: {
        color: '#000000',
        fontSize: 14,
        fontWeight: '600',
        textTransform: 'capitalize'
    },
    other_text: {
        color: '#828282',
        textAlign: 'center',
        marginTop: 15,
        marginBottom: 12,
    },
    other_button: {
        flexDirection: 'row',
        width: '100%',
        justifyContent: 'space-between',
        marginBottom: 10,
    },
    icon_button: {
        borderRadius: 7,
        marginTop: 5,
        backgroundColor: '#F1F5F9',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        paddingVertical: 10,
        marginBottom: 10,
        width: '48%',
    },
    icon_button_text: {
        color: '#000000',
        fontSize: 14,
        fontWeight: '400'
    }
});

export default RegistrationScreen;
