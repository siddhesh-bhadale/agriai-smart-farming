import React, { useState, useEffect } from 'react';
import { View, Text, StyleSheet, Image, ScrollView, SafeAreaView, TouchableOpacity, ActivityIndicator, Dimensions, PermissionsAndroid, Platform, Alert, LogBox } from 'react-native';
import { useNavigation } from '@react-navigation/native';
import AsyncStorage from '@react-native-async-storage/async-storage';
import BottomNavigation from '../navigation/BottomNavigation';
import Ionicons from 'react-native-vector-icons/Ionicons'
import Fontisto from 'react-native-vector-icons/Fontisto'
import Feather from 'react-native-vector-icons/Feather'
import MaterialCommunityIcons from 'react-native-vector-icons/MaterialCommunityIcons'
import MaterialIcons from 'react-native-vector-icons/MaterialIcons'
import axios from 'axios';
import Config from 'react-native-config';
import Geolocation from 'react-native-geolocation-service';
import { check, request, PERMISSIONS, RESULTS } from 'react-native-permissions';

LogBox.ignoreAllLogs();
const HomeScreen = () => {
  const navigation = useNavigation();
  const { width, height } = Dimensions.get('window');
  const [userData, setUserData] = useState(null);
  const [weatherData, setWeatherData] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const [latitude, setlatitude] = useState('18.516726');
  const [longtitude, setlongtitude] = useState('73.856255');
  let rainfall = 0.0;

  const kelvinToCelsius = kelvin => (kelvin - 273.15).toFixed(1);

  useEffect(() => {
    handleLocationPermission()
  }, [latitude, longtitude]);

  const fetchData = async () => {
    setIsLoading(true);
    try {
      // Load user data
      const storedUserData = await AsyncStorage.getItem('userData');
      setUserData(JSON.parse(storedUserData));

      // Load weather data
      const response = await axios.get(
        `http://api.openweathermap.org/data/2.5/weather?lat=${latitude}&lon=${longtitude}&appid=${Config.API_KEY}`
      );

      // Parse rainfall data
      if (response.data.rain) {
        rainfall = response.data.rain['1h'];
      }

      // Set weather data
      setWeatherData({
        ...response.data,
        rainfall: rainfall
      });

      setIsLoading(false);
    } catch (error) {
      console.error('Error fetching data:', error);
      setIsLoading(false);
    }
  };


  // ask permission for location and get location 

  const requestLocationPermission = async () => {
    if (Platform.OS === 'ios') {
      const status = await check(PERMISSIONS.IOS.LOCATION_WHEN_IN_USE);

      switch (status) {
        case RESULTS.UNAVAILABLE:
          console.log('This feature is not available on this device');
          return false;
        case RESULTS.DENIED:
          const requestStatus = await request(PERMISSIONS.IOS.LOCATION_WHEN_IN_USE);
          return requestStatus === RESULTS.GRANTED;
        case RESULTS.LIMITED:
        case RESULTS.GRANTED:
          return true;
        case RESULTS.BLOCKED:
          console.log('The permission is denied and not requestable anymore');
          return false;
      }
    } else if (Platform.OS === 'android') {
      const status = await PermissionsAndroid.check(PermissionsAndroid.PERMISSIONS.ACCESS_FINE_LOCATION);

      if (status) {
        return true;
      } else {
        const requestStatus = await PermissionsAndroid.request(PermissionsAndroid.PERMISSIONS.ACCESS_FINE_LOCATION);
        return requestStatus === PermissionsAndroid.RESULTS.GRANTED;
      }
    }
  };

  const getLocation = () => {
    Geolocation.getCurrentPosition(
      (position) => {
        // console.log(position.coords.latitude, " ", position.coords.longitude);
        setlatitude(position.coords.latitude)
        setlongtitude(position.coords.longitude)
      },
      (error) => {
        console.error(error);
      },
      {
        enableHighAccuracy: true,
        timeout: 15000,
        maximumAge: 10000,
      }
    );
  };

  const handleLocationPermission = async () => {
    const isGranted = await requestLocationPermission();
    if (isGranted) {
      getLocation();
      fetchData();
    } else {
      console.log('Location permission denied');
      // Optionally, re-ask for permission or guide the user to settings
    }
  };


  return (
    <SafeAreaView style={{ height: height, backgroundColor: '#ffffff' }}>
      {isLoading ? (
        <View style={{ height: height - 60, alignItems: 'center', justifyContent: 'center' }}>
          <ActivityIndicator style={styles.loader} size="large" color="#46c67c" />
        </View>
      ) : (
        <ScrollView>
          <View style={styles.mainContainer}>
            <Image style={styles.backgroundImage} source={require('../assets/login/homebg.png')} />
            <View style={styles.container}>
              <View style={styles.userInfo}>
                <View style={styles.userInfoLeft}>
                  <Text style={styles.userName}>Welcome, {userData.name}</Text>
                  <Text style={styles.weather}>It's Sunny Day</Text>
                </View>
                <View style={styles.userInfoRight}>
                  <TouchableOpacity style={styles.trakCropButton}>
                    <Text style={styles.trakCropButtonText}>Track Crop</Text>
                  </TouchableOpacity>
                </View>
              </View>

              <View style={styles.weatherInfo}>
                <View style={styles.weatherWrapper}>
                  <View style={styles.weatherInfoLeft}>
                    <View style={styles.weatherItem}>
                      <View style={styles.itemLeft}>
                        <View style={styles.circule}>
                          <Ionicons name="thermometer-outline" size={23} color="#ffffff" />
                        </View>
                      </View>
                      <View style={styles.itemRight}>
                        <Text style={styles.wetherStack}>{kelvinToCelsius(weatherData.main.temp ? weatherData.main.temp : '0')}°C</Text>
                        <Text style={styles.weatherName}>Temprature</Text>
                      </View>
                    </View>
                    <View style={styles.weatherItem}>
                      <View style={styles.itemLeft}>
                        <View style={[styles.circule, { backgroundColor: '#9874cc' }]}>
                          <Feather name="wind" size={23} color="#ffffff" />
                        </View>
                      </View>
                      <View style={styles.itemRight}>
                        <Text style={styles.wetherStack}>{weatherData.wind.speed ? weatherData.wind.speed : '0'}m/s</Text>
                        <Text style={styles.weatherName}>Wind Speed</Text>
                      </View>
                    </View>
                  </View>
                  <View style={styles.weatherInfoLeft}>
                    <View style={styles.weatherItem}>
                      <View style={styles.itemLeft}>
                        <View style={[styles.circule, { backgroundColor: '#21a0c3' }]}>
                          <Fontisto name="rain" size={23} color="#ffffff" />
                        </View>
                      </View>
                      <View style={styles.itemRight}>
                        <Text style={styles.wetherStack}>{rainfall ? rainfall : '0.0'} mm</Text>
                        <Text style={styles.weatherName}>Rainfall</Text>
                      </View>
                    </View>
                    <View style={styles.weatherItem}>
                      <View style={styles.itemLeft}>
                        <View style={[styles.circule, { backgroundColor: '#e5be28' }]}>
                          <Ionicons name="water-outline" size={23} color="#ffffff" />
                        </View>
                      </View>
                      <View style={styles.itemRight}>
                        <Text style={styles.wetherStack}>{weatherData.main.humidity ? weatherData.main.humidity : '0'}%</Text>
                        <Text style={styles.weatherName}>Humidity</Text>
                      </View>
                    </View>
                  </View>
                </View>
              </View>

              <TouchableOpacity style={styles.detect} onPress={() => { navigation.navigate('DiseaseDetection') }}>
                <View style={styles.dtectLeft}>
                  <MaterialCommunityIcons name="line-scan" size={24} color="#ffffff" />
                  <Text style={styles.ditectText}>Detect Diseases of Crops</Text>
                </View>
                <MaterialIcons name="keyboard-arrow-right" size={24} color="#ffffff" />
              </TouchableOpacity>

              <View style={styles.cropRecommendationContainer}>
                <Text style={styles.recomendationText}>Crop Reccomentation</Text>
                <ScrollView horizontal={true} style={styles.cropRecommendationWrapper}>
                  <TouchableOpacity style={styles.recomendationImage}>
                    <Image style={styles.recommendationImage} source={require('../assets/home/tomato.png')} />
                  </TouchableOpacity>
                  <TouchableOpacity style={styles.recomendationImage}>
                    <Image style={styles.recommendationImage} source={require('../assets/home/corn.png')} />
                  </TouchableOpacity>
                  <TouchableOpacity style={styles.recomendationImage}>
                    <Image style={styles.recommendationImage} source={require('../assets/home/onion.png')} />
                  </TouchableOpacity>
                  <TouchableOpacity style={styles.recomendationImage}>
                    <Image style={styles.recommendationImage} source={require('../assets/home/tomato.png')} />
                  </TouchableOpacity>
                </ScrollView>
              </View>

              <View style={styles.traindingDiseaseContainer}>
                <Text style={styles.traindingText}>Trending Diseases</Text>
                <ScrollView horizontal={true} style={styles.traindingDiseaseWrapper}>
                  <TouchableOpacity style={styles.traindingItem}>
                    <View style={styles.traindingImageContainer}>
                      <Image style={styles.traindingImage} source={require('../assets/home/tomato.png')} />
                    </View>
                    <View style={styles.traindingItemRight}>
                      <Text style={styles.traindingItemName}>African Mole Cricket</Text>
                      <Text style={styles.traindingItemType}>Insects</Text>
                    </View>
                  </TouchableOpacity>
                  <TouchableOpacity style={styles.traindingItem}>
                    <View style={styles.traindingImageContainer}>
                      <Image style={styles.traindingImage} source={require('../assets/home/tomato.png')} />
                    </View>
                    <View style={styles.traindingItemRight}>
                      <Text style={styles.traindingItemName}>Alternaria solani</Text>
                      <Text style={styles.traindingItemType}>fungal</Text>
                    </View>
                  </TouchableOpacity>
                  <TouchableOpacity style={styles.traindingItem}>
                    <View style={styles.traindingImageContainer}>
                      <Image style={styles.traindingImage} source={require('../assets/home/tomato.png')} />
                    </View>
                    <View style={styles.traindingItemRight}>
                      <Text style={styles.traindingItemName}>Septoria</Text>
                      <Text style={styles.traindingItemType}>fungi</Text>
                    </View>
                  </TouchableOpacity>
                </ScrollView>
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
  },
  backgroundImage: {
    position: 'absolute',
    top: -40
  },
  container: {
    // position: 'relative',
    // top: -310,
    marginTop: 70,
  },
  userInfo: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    flexDirection: 'row',
    paddingHorizontal: 25,
  },
  userInfoLeft: {
    display: 'flex',
    alignItems: 'flex-start',
    justifyContent: 'center',
  },
  userName: {
    color: '#000000',
    fontSize: 17,
    fontWeight: '600',
  },
  weather: {
    color: '#000000',
    fontSize: 16,
    fontWeight: '300',
  },
  trakCropButton: {
    backgroundColor: '#46c67c',
    paddingVertical: 8,
    paddingHorizontal: 20,
    borderRadius: 5
  },
  trakCropButtonText: {
    color: '#ffffff'
  },

  //wether info
  weatherInfo: {
    backgroundColor: '#ffffff',
    marginHorizontal: 25,
    paddingVertical: 30,
    paddingHorizontal: 20,
    borderRadius: 5,
    marginTop: 30,
    shadowColor: "#999797",
    shadowOffset: {
      width: 0,
      height: 12,
    },
    shadowOpacity: 0.58,
    shadowRadius: 16.00,

    elevation: 24,
  },
  weatherWrapper: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    flexDirection: 'row',
  },
  weatherInfoLeft: {
    display: 'flex',
    alignItems: 'flex-start',
    justifyContent: 'center',
    gap: 14,
    paddingRight: 5
  },

  weatherItem: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    flexDirection: 'row',
    gap: 7
  },
  circule: {
    paddingVertical: 8,
    paddingHorizontal: 8,
    backgroundColor: '#46c67c',
    borderRadius: 100
  },
  wetherStack: {
    color: '#000000',
    fontSize: 16,
    fontWeight: '600'
  },
  weatherName: {
    color: '#000000',
    fontSize: 13,
    fontWeight: '300'
  },

  // detect Button
  detect: {
    backgroundColor: '#46c67c',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    flexDirection: 'row',
    marginHorizontal: 25,
    paddingVertical: 15,
    paddingHorizontal: 15,
    marginVertical: 28,
    borderRadius: 7
  },
  dtectLeft: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    flexDirection: 'row',
    gap: 10
  },
  ditectText: {
    color: '#ffffff',
    fontSize: 15,
    fontWeight: '600',
  },

  // crop Recommendation 
  cropRecommendationContainer: {
    marginHorizontal: 25
  },
  recomendationText: {
    color: '#000000',
    fontSize: 16,
    fontWeight: '600'
  },
  cropRecommendationWrapper: {
    marginTop: 12
  },
  recomendationImage: {
    marginRight: 10,
    borderRadius: 7
  },
  recommendationImage: {
    borderRadius: 7
  },

  // traindng diseases
  traindingDiseaseContainer: {
    marginHorizontal: 25,
    marginVertical: 20,
  },
  traindingText: {
    color: '#000000',
    fontSize: 16,
    fontWeight: '600'
  },
  traindingDiseaseWrapper: {
    marginTop: 5,
    paddingVertical: 7,
  },
  traindingItem: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    flexDirection: 'row',
    gap: 15,
    backgroundColor: '#ffffff',
    borderWidth: 1,
    borderColor: '#e8e8e8',
    marginRight: 10,
    paddingVertical: 10,
    paddingHorizontal: 15,
    borderRadius: 7,

    // shadowColor: "#999797",
    // shadowOffset: {
    //   width: 0,
    //   height: 12,
    // },
    // shadowOpacity: 0.1,
    // shadowRadius: 16.00,

    // elevation: 24,
  },
  traindingItemRight: {
    display: 'flex',
    alignItems: 'flex-start',
    justifyContent: 'center',
    gap: 10,
    width:150
  },
  traindingImageContainer: {
    borderRadius: 7
  },
  traindingImage: {
    borderRadius: 7
  },
  traindingItemName: {
    color: '#000000',
    fontSize: 16,
    fontWeight: '600'
  },
  traindingItemType: {
    paddingVertical: 5,
    paddingHorizontal: 10,
    borderRadius: 5,
    color: '#ffffff',
    backgroundColor: '#46c67c',
  }

});

export default HomeScreen;

// import { SafeAreaView, StyleSheet, Text, View } from 'react-native'
// import React from 'react'
// import BottomNavigation from '../navigation/BottomNavigation'

// const HomeScreen = () => {
//   return (
//     <SafeAreaView>
//       <BottomNavigation/>
//     </SafeAreaView>
//   )
// }

// export default HomeScreen

// const styles = StyleSheet.create({})