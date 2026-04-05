import React, { useState, useEffect } from 'react';
import { StyleSheet, Text, View, SafeAreaView, ScrollView, PermissionsAndroid, } from 'react-native';
import { NavigationContainer } from '@react-navigation/native';
import { createNativeStackNavigator } from '@react-navigation/native-stack';
import RegistrationScreen from './src/components/register/RegistrationScreen';
import LoginScreen from './src/components/register/LoginScreen';
import HomeScreen from './src/screens/HomeScreen';
import Auth from './src/components/register/auth';
import FarmerToBuyer from './src/screens/FarmerToBuyer';
import FarmerToFarmer from './src/screens/FarmerToFarmer';
import User from './src/screens/User';
import DiseaseDetection from './src/screens/DiseaseDetection';
import Profile from './src/components/farmerToFarmer/Profile';
import Message from './src/components/farmerToFarmer/Message';
import AddFarmer from './src/components/farmerToFarmer/AddFarmer';
import PostForm from './src/components/farmerToFarmer/PostForm';
import DiseaseDetails from './src/components/diseaseDetection/DiseaseDetails';
import ChangePassword from './src/components/register/ChangePassword';
import ForgotPasswordScreen from './src/components/register/ForgotPasswordScreen';
import FriendRequests from './src/components/farmerToFarmer/FriendRequests';
import Chat from './src/components/farmerToFarmer/components/Chat';
import UpdateProfileForm from './src/components/farmerToFarmer/components/UpdateProfileForm ';

const Stack = createNativeStackNavigator();

function App() {

  

  return (
    <NavigationContainer>
      <Stack.Navigator initialRouteName="Auth" screenOptions={{ headerShown: false }}>
        <Stack.Screen name="Auth" component={Auth} />
        <Stack.Screen name="Login" component={LoginScreen} />
        <Stack.Screen name="Registration" component={RegistrationScreen} />
        <Stack.Screen name="Home" component={HomeScreen} />
        <Stack.Screen name="FarmerToBuyer" component={FarmerToBuyer} />
        <Stack.Screen name="FarmerToFarmer" component={FarmerToFarmer} />
        <Stack.Screen name="User" component={User} />
        <Stack.Screen name="DiseaseDetection" component={DiseaseDetection} />
        <Stack.Screen name="DiseaseDetails" component={DiseaseDetails} />
        <Stack.Screen name="Profile" component={Profile} />
        <Stack.Screen name="Message" component={Message} />
        <Stack.Screen name="AddFarmer" component={AddFarmer} />
        <Stack.Screen name="FriendRequests" component={FriendRequests} />
        <Stack.Screen name="PostForm" component={PostForm} />
        <Stack.Screen name="ChangePassword" component={ChangePassword} />
        <Stack.Screen name="ForgotPasswordScreen" component={ForgotPasswordScreen} />
        <Stack.Screen name="Chat" component={Chat} />
        <Stack.Screen name="UpdateProfileForm" component={UpdateProfileForm} />
      </Stack.Navigator>
    </NavigationContainer>
  );
}

const styles = StyleSheet.create({

});

export default App;