import React from 'react';
import { Text, TouchableOpacity, View, StyleSheet } from 'react-native';
import { useNavigation } from '@react-navigation/native';
import Icon from 'react-native-vector-icons/FontAwesome'
import Ionicons from 'react-native-vector-icons/Ionicons'
import MaterialCommunityIcons from 'react-native-vector-icons/MaterialCommunityIcons'
import AntDesign from 'react-native-vector-icons/AntDesign'
import Fontisto from 'react-native-vector-icons/Fontisto'
import FontAwesome from 'react-native-vector-icons/FontAwesome'


function BottomNavigation() {
  const navigation = useNavigation();
  return (
    <View style={styles.main_Container}>
      <View style={styles.block}></View>
      <View style={styles.menu_container}>
        <TouchableOpacity style={styles.button} onPress={() => navigation.navigate('Home')}>
          <AntDesign name="home" size={23} color="#3f484b" />
          <Text style={styles.lable}>Home</Text>
        </TouchableOpacity>
        <TouchableOpacity style={styles.button} onPress={() => navigation.navigate('FarmerToBuyer')}>
          <AntDesign name="appstore-o" size={23} color="#3f484b" />
          <Text style={styles.lable}>Sell</Text>
        </TouchableOpacity>
        <TouchableOpacity style={styles.scan} onPress={() => navigation.navigate('DiseaseDetection')}>
          <MaterialCommunityIcons name="line-scan" size={24} color="#3f484b" />
        </TouchableOpacity>
        <TouchableOpacity style={styles.button} onPress={() => navigation.navigate('FarmerToFarmer')}>
          <Fontisto name="world-o" size={23} color="#3f484b" />
          <Text style={styles.lable}>New</Text>
        </TouchableOpacity>
        <TouchableOpacity style={styles.button} onPress={() => navigation.navigate('User')}>
          <FontAwesome name="user-o" size={23} color="#3f484b" />
          <Text style={styles.lable}>You</Text>
        </TouchableOpacity>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  main_Container: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    width: '100%',
    // borderWidth:1
  },
  block: {
    // borderWidth:1,
    // borderColor:'red',
    width: '100%',
    height: 60
  },
  menu_container: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    paddingHorizontal: 22,
    position: 'absolute',
    width: '100%',
    alignSelf: 'center',
    backgroundColor: '#e9f1f1',
    zIndex: 100,
    bottom: 0,
    // left: 0,
    borderTopWidth: 0.2,
    borderColor: '#e0e0e0',
  },
  button: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    paddingHorizontal: 10,
    // maxWidth: 77,
    // borderWidth: 1
  },
  lable: {
    fontSize: 11,
    color: '#3f484b',
    fontWeight:'400',
    marginTop:3
  },
  scan: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    borderWidth: 4,
    borderColor: '#ffffff',
    borderRadius: 100,
    paddingHorizontal: 14,
    paddingVertical: 13,
    position: 'relative',
    top: -25,
    backgroundColor: '#e9f1f1',
  }
})

export default BottomNavigation;