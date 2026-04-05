import React, { useState } from 'react';
import {
  View,
  Text,
  Image,
  TouchableOpacity,
  StyleSheet,
  Platform,
  Dimensions,
  SafeAreaView,
  ScrollView,
  ActivityIndicator,
} from 'react-native';
import axios from 'axios';
import { launchCamera, launchImageLibrary } from 'react-native-image-picker';
import PermissionsService, { isIOS } from '../utils/Permissions';
import Config from 'react-native-config';
import MaterialCommunityIcons from 'react-native-vector-icons/MaterialCommunityIcons';
import Ionicons from 'react-native-vector-icons/Ionicons';
import BottomNavigation from '../navigation/BottomNavigation';

const DiseaseDetection = ({ navigation }) => {
  const [image, setImage] = useState('');
  const [label, setLabel] = useState('');
  const [result, setResult] = useState('');
  const [loading, setLoading] = useState(false);

  const options = {
    mediaType: 'photo',
    quality: 1,
    width: 256,
    height: 256,
    includeBase64: true,
  };

  const manageCamera = async type => {
    try {
      if (!(await PermissionsService.hasCameraPermission())) {
        return [];
      } else {
        if (type === 'Camera') {
          openCamera();
        } else {
          openLibrary();
        }
      }
    } catch (err) {
      console.log(err);
    }
  };

  const openCamera = async () => {
    launchCamera(options, async response => {
      if (response.didCancel) {
        console.log('User cancelled image picker');
      } else if (response.error) {
        console.log('ImagePicker Error: ', response.error);
      } else if (response.customButton) {
        console.log('User tapped custom button: ', response.customButton);
      } else {
        const uri = response?.assets[0]?.uri;
        const path = Platform.OS !== 'ios' ? uri : 'file://' + uri;
        handleImage(path, response);
      }
    });
  };

  // const openLibrary = async () => {
  //   launchImageLibrary(options, async response => {
  //     if (response.didCancel) {
  //       console.log('User cancelled image picker');
  //     } else if (response.error) {
  //       console.log('ImagePicker Error: ', response.error);
  //     } else if (response.customButton) {
  //       console.log('User tapped custom button: ', response.customButton);
  //     } else {
  //       const uri = response.assets[0].uri;
  //       const path = Platform.OS !== 'ios' ? uri : 'file://' + uri;
  //       handleImage(path, response);
  //     }
  //   });
  // };

  const pickImage = () => {
    launchImageLibrary({ mediaType: 'photo' }, (response) => {
      if (response.didCancel) {
        console.log('User cancelled image picker');
      } else if (response.error) {
        console.log('ImagePicker Error: ', response.error);
      } else {
        const uri = response.assets[0].uri;
        const path = Platform.OS !== 'ios' ? uri : 'file://' + uri;
        handleImage(path, response);
      }
    });
  };

  const handleImage = (path, response) => {
    if (response.didCancel) {
      console.log('User cancelled image picker');
    } else if (response.error) {
      console.log('ImagePicker Error: ', response.error);
    } else {
      const uri = path;
      const name = response.assets[0]?.fileName;
      const type = response.assets[0]?.type;

      if (!uri || !name || !type) {
        console.log('Invalid image data');
        return;
      }

      const formData = new FormData();
      formData.append('file', {
        uri,
        name,
        type,
      });

      sendImageToServer(formData);
      setImage(path);
    }
  };


  const sendImageToServer = async (formData) => {
    setLoading(true);
    try {
      const response = await axios.post(`${Config.MODEL_API_URL}/predict`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      if (response.data && response.data.predictions) {
        setLabel(response.data.predictions);
        setLoading(false);
      } else {
        setLabel('Failed to predict');
        setLoading(false);
      }
    } catch (error) {
      console.error('Error sending image:', error);
      setLabel('Failed to predict');
      setLoading(false);
    }
  };

  return (
    <SafeAreaView style={{ height: '100%' }}>
      {loading ? (
        <View style={styles.loader}>
          <ActivityIndicator size="large" color="#46c67c" />
          <Text style={{ color: '#000', marginTop: 5 }}>Predicting...</Text>
        </View>
      ) : (
        <ScrollView>
          {image ? (
            <View>
              <Image source={{ uri: image }} style={styles.backgroundImage} />

              <View style={styles.container2}>
                {label && <Text style={styles.result}>Crop Name : <Text style={{ fontWeight: '400' }}>{label.type}</Text></Text>}
                {label && <Text style={styles.result}>Crop Disease : <Text style={{ fontWeight: '400' }}>{label.name}</Text></Text>}
                <View style={styles.buttonWrapper}>
                  <TouchableOpacity style={[styles.moreDetailsButton, { backgroundColor: '#ffffff', borderWidth: 1, borderColor: '#e0e0e0' }]} onPress={openCamera}>
                    <Text style={[styles.moreDetailsButtonText, { color: '#000' }]}>Take Photo</Text>
                  </TouchableOpacity>
                  <TouchableOpacity style={styles.moreDetailsButton} onPress={() => { navigation.navigate({ name: 'DiseaseDetails', params: { result: label }, merge: true, }) }}>
                    <Text style={styles.moreDetailsButtonText}>More Details</Text>
                  </TouchableOpacity>
                </View>
              </View>
            </View>
          ) : (
            <View>
              <Image source={require('../assets/diseaseDetection/backgroundImage.png')} style={styles.backgroundImage} />

              <View style={styles.container}>
                <Text style={styles.heading}>Scanner</Text>

                <TouchableOpacity onPress={openCamera}>
                  <Image source={require('../assets/diseaseDetection/scan-line.png')} style={styles.scanImage} />
                </TouchableOpacity>

                <View style={styles.buttonContainer}>
                  <TouchableOpacity style={styles.button} onPress={openCamera}>
                    <MaterialCommunityIcons name="camera" size={32} color="#46c67c" />
                  </TouchableOpacity>
                  <TouchableOpacity style={styles.button} onPress={pickImage}>
                    <Ionicons name="image" size={32} color="#46c67c" />
                  </TouchableOpacity>
                </View>
              </View>
            </View>
          )}
        </ScrollView>
      )}
      <BottomNavigation/>
    </SafeAreaView>
  );
};

const styles = StyleSheet.create({
  loader: {
    borderWidth: 1,
    height: '100%',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
  },
  backgroundImage: {
    flex: 1,
    resizeMode: 'cover',
    width: Dimensions.get('window').width,
    height: Dimensions.get('window').height
  },
  container: {
    position: 'absolute',
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    borderWidth: 1,
    width: Dimensions.get('window').width,
    height: Dimensions.get('window').height
  },
  heading: {
    fontSize: 23,
    fontWeight: '600',
    position: 'absolute',
    top: 70,
    color: '#ffffff'
  },
  scanImage: {

  },
  instructions: {
    fontSize: 18,
    marginBottom: 20,
  },
  buttonContainer: {
    flexDirection: 'row'
  },
  button: {
    backgroundColor: '#ffffff',
    paddingHorizontal: 15,
    paddingVertical: 10,
    borderRadius: 7,
    margin: 10,
  },
  buttonText: {
    color: 'white',
    fontSize: 16,
    fontWeight: 'bold',
  },

  container2: {
    position: 'absolute',
    bottom: 0,
    backgroundColor: '#ffffff',
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    width: Dimensions.get('window').width,
    paddingHorizontal: 15,
    paddingBottom: 40,
    paddingTop: 30,
    borderTopLeftRadius: 30,
    borderTopRightRadius: 30
  },
  result: {
    fontSize: 16,
    fontWeight: '600',
    marginBottom: 4,
    color: '#000000',
    textAlign: 'center'
  },
  buttonWrapper: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    width: '100%',
    marginTop: 25
  },
  moreDetailsButton: {
    width: '49%',
    backgroundColor: '#46c67c',
    borderRadius: 5,
    paddingVertical: 10,
  },
  moreDetailsButtonText: {
    color: 'white',
    fontSize: 14,
    fontWeight: 'bold',
    alignSelf: 'center'
  },
});

export default DiseaseDetection;
