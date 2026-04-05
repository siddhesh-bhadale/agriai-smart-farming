import React, { useState } from 'react';
import { View, Text, TextInput, SafeAreaView, ToastAndroid, ActivityIndicator, StyleSheet, Image, TouchableOpacity, ScrollView } from 'react-native';
import { useNavigation, RouteProp } from '@react-navigation/native';
import { StackNavigationProp } from '@react-navigation/stack';
import AsyncStorage from '@react-native-async-storage/async-storage';
import axios from 'axios';
import { validateEmail, validatePassword } from '../../utils/utils';
import Config from 'react-native-config'

const LoginScreen = () => {
    const navigation = useNavigation();
    const [email, setEmail] = useState('');
    const [password, setPassword] = useState('');
    const [loading, setLoading] = useState(false);

    const showToast = (message) => {
        ToastAndroid.showWithGravityAndOffset(
            message,
            ToastAndroid.LONG,
            ToastAndroid.BOTTOM,
            25,
            50
        );
    };

    const handleLogin = async () => {
        if (!validateEmail(email) || !validatePassword(password)) {
            showToast('Invalid Input', 'Please enter valid email and password.');
            return;
        }

        setLoading(true);

        try {
            const response = await axios.post(`${Config.API_URL}/login`, {
                email,
                password,
            });

            if (response && response.data) {
                const { userData } = response.data;
                try {
                    const jsonValue = JSON.stringify(userData);
                    await AsyncStorage.setItem('userData', jsonValue);
                    showToast('Login Succesfull!');
                    setLoading(false);
                    navigation.reset({
                        index: 0,
                        routes: [{ name: 'Auth' }],
                    });
                } catch (e) {
                    console.log(e)
                }
            } else {
                setLoading(false);
                showToast('Login Failed', 'Invalid response from the server');
            }
        } catch (error) {
            setLoading(false);
            console.log(error);
            showToast('Login Failed', error.response ? error.response.data.message : 'An error occurred');
        }
    };


    return (
        <SafeAreaView>
            <ScrollView>
                <View style={styles.mainContainer}>
                    <Image style={styles.backgroundImage} source={require('../../assets/login/Subtract.png')} />
                    <Image style={styles.logo} source={require('../../assets/logo/logo.png')} />
                    <View style={styles.bottom_section}>
                        <Text style={styles.heading}>Login</Text>
                        <View style={styles.form}>
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
                                    placeholder='Enter your password'
                                    placeholderTextColor={"#828282"}
                                    value={password}
                                    autoCapitalize='none'
                                    autoCorrect={false}
                                    onChangeText={setPassword}
                                    secureTextEntry
                                />
                            </View>

                            <TouchableOpacity style={styles.forgotPassword} onPress={() => { navigation.navigate('ForgotPasswordScreen') }}>
                                <Text style={styles.forgotPassword_text}>Forgot Password?</Text>
                            </TouchableOpacity>

                            <View style={styles.button_wrapper}>
                                {loading ? (
                                    <ActivityIndicator size="large" color="#46c67c" />
                                ) : (
                                    <TouchableOpacity style={styles.button} onPress={handleLogin}>
                                        <Text style={styles.button_text}>Login</Text>
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
                                    <Text style={styles.text2}>Don't have an account? </Text>
                                    <TouchableOpacity style={styles.button2} onPress={() => navigation.navigate('Registration')}>
                                        <Text style={styles.button_text2}>Registration</Text>
                                    </TouchableOpacity>
                                </View>
                            </View>
                        </View>
                    </View>
                </View>
            </ScrollView>
        </SafeAreaView>
    );
};

const styles = StyleSheet.create({
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
        top: -100,
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

    forgotPassword: {
        marginTop: -15,
        alignItems: 'flex-end',
        marginBottom: 10
    },
    forgotPassword_text: {
        color: '#3f484b'
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

export default LoginScreen;
