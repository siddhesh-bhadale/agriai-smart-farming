import React, { useState, useEffect } from 'react';
import { View, TextInput, Button, ActivityIndicator, TouchableOpacity, ToastAndroid, StyleSheet, Text, SafeAreaView, ScrollView, Image } from 'react-native';
import axios from 'axios';
import AsyncStorage from '@react-native-async-storage/async-storage';
import Ionicons from 'react-native-vector-icons/Ionicons';
import Config from 'react-native-config';

const ChangePassword = ({ navigation }) => {
    const API_BASE_URL = 'https://digital-wellbing-api.onrender.com';
    const [email, setEmail] = useState('');
    const [otp, setOTP] = useState('');
    const [newPassword, setNewPassword] = useState('');
    const [loading, setLoading] = useState(false);
    const [isLoading, setIsLoading] = useState(true);
    const [otpSend, setOtpSend] = useState(false);

    useEffect(() => {
        const timeout = setTimeout(() => {
            setIsLoading(false);
        }, 500);

        return () => clearTimeout(timeout); // Clear the timeout on component unmount
    }, []);

    useEffect(() => {
        const getEmail = async () => {
            try {
                const userId = await AsyncStorage.getItem('userData');
                const userData = JSON.parse(userId);
                setEmail(userData.email);
            } catch (error) {
                console.error("Error retrieving user data:", error);
            }
        };

        getEmail();
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

    const handleForgotPassword = async () => {
        setLoading(true);

        try {
            const response = await axios.post(`${Config.API_URL}/forgot-password`, { email });
            setOtpSend(true)
            showToast(response.data.message);
        } catch (error) {
            showToast(error.response.data.message);
        } finally {
            setLoading(false);
        }
    };

    const handleResetPassword = async () => {
        setLoading(true);

        try {
            const response = await axios.post(`${Config.API_URL}/reset-password`, { email, otp, newPassword });
            showToast(response.data.message);
            navigation.goBack()
            // Navigate to login screen or any other appropriate action after successful password reset
        } catch (error) {
            showToast(error.response.data.message);
        } finally {
            setLoading(false);
        }
    };

    return (
        <SafeAreaView style={{ backgroundColor: '#ffffff', height: '100%' }}>
            {isLoading ? (
                <ActivityIndicator style={styles.loader} size="large" color="#46c67c" />
            ) : (
                <ScrollView>
                    <View style={styles.mainContainer}>
                        <View style={styles.topStatusBar}>
                            <TouchableOpacity style={styles.backButton} onPress={() => navigation.navigate('FarmerToFarmer')}>
                                <Ionicons name="chevron-back-outline" size={22} color="#585C60" />
                            </TouchableOpacity>
                            <Text style={styles.heading}>Forgot Password</Text>
                        </View>

                        {!otpSend ? (
                            <View style={styles.section}>
                                <View style={styles.imageContainer}>
                                    <Image style={styles.Image} source={require('../../assets/icon/email.png')} />
                                </View>

                                <Text style={styles.mesageText}>Please Enter Your Email Address To Recive Verification OTP</Text>

                                <View style={styles.form}>
                                    <View>
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
                                        {loading ? (
                                            <ActivityIndicator size="large" color="#46c67c" />
                                        ) : (
                                            <TouchableOpacity style={styles.button} onPress={handleForgotPassword}>
                                                <Text style={styles.button_text}>Send OTP</Text>
                                            </TouchableOpacity>
                                        )}
                                        {/* <Button title="Forgot Password" onPress={handleForgotPassword} /> */}
                                    </View>
                                </View>
                            </View>
                        ) : (
                            <View style={styles.section}>
                                <View style={styles.imageContainer}>
                                    <Image style={styles.Image} source={require('../../assets/icon/password.png')} />
                                </View>

                                <Text style={styles.mesageText}>Please Enter The 6 Digit Code Send To {email}</Text>
                                <View style={styles.form}>
                                    <View style={styles.input_section}>
                                        <Text style={styles.text}>Enter OTP</Text>
                                        <TextInput
                                            style={styles.input}
                                            placeholder='Enter OTP'
                                            placeholderTextColor={"#828282"}
                                            value={otp}
                                            autoCapitalize='none'
                                            autoCorrect={false}
                                            onChangeText={setOTP}
                                        />
                                    </View>
                                    <View style={styles.input_section}>
                                        <Text style={styles.text}>Enter new password</Text>
                                        <TextInput
                                            style={styles.input}
                                            placeholder='Enter new password'
                                            placeholderTextColor={"#828282"}
                                            value={newPassword}
                                            autoCapitalize='none'
                                            autoCorrect={false}
                                            onChangeText={setNewPassword}
                                        />
                                    </View>
                                    <TouchableOpacity style={styles.resendOtp} onPress={handleForgotPassword}>
                                        <Text style={styles.resendOtp_text}>Resend OTP</Text>
                                    </TouchableOpacity>
                                    {/* <TextInput
                                    placeholder="Enter OTP"
                                    value={otp}
                                    onChangeText={setOTP}
                                    style={{ color: '#000000', marginBottom: 10, paddingHorizontal: 10, borderWidth: 1, borderColor: 'gray', borderRadius: 5 }}
                                /> */}
                                    {/* <TextInput
                                    placeholder="Enter new password"
                                    value={newPassword}
                                    onChangeText={setNewPassword}
                                    style={{ color: '#000000', marginBottom: 10, paddingHorizontal: 10, borderWidth: 1, borderColor: 'gray', borderRadius: 5 }}
                                /> */}
                                    {/* <Button title="Reset Password" onPress={handleResetPassword} /> */}

                                    {loading ? (
                                        <ActivityIndicator size="large" color="#46c67c" />
                                    ) : (
                                        <TouchableOpacity style={styles.button} onPress={handleResetPassword}>
                                            <Text style={styles.button_text}>Reset Password</Text>
                                        </TouchableOpacity>
                                    )}
                                </View>


                                {/* {loading && <ActivityIndicator style={{ marginTop: 20 }} />} */}
                            </View>
                        )}
                    </View>
                </ScrollView>
            )}
        </SafeAreaView >
    );
};

const styles = StyleSheet.create({
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
        paddingVertical: 16,
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
        fontWeight: '600',
        textAlign: 'center',
        width: '75%'
    },
    imageContainer: {
        width: 230,
        height: 230,
        borderRadius: 120,
        // borderWidth:1,
        alignItems: 'center',
        justifyContent: 'center',
        alignSelf: 'center',
        marginVertical: 20,
        backgroundColor: '#D7FFE8'
    },
    Image: {
        width: '40%',
        height: '40%'
    },
    mesageText: {
        color: '#000000',
        textAlign: 'center',
        width: '80%',
        alignSelf: 'center',
        fontSize: 13,
        lineHeight: 21,
        fontWeight: '400',
        // textTransform:'capitalize',
        marginTop: 23,
        marginBottom: 40
    },
    form: {
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
    resendOtp: {
        alignItems: 'center',
        marginBottom: 20
    },
    resendOtp_text: {
        color: '#274ae6'
    }
})

export default ChangePassword;
