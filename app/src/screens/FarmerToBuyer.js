import {
  SafeAreaView,
  ScrollView,
  StyleSheet,
  Text,
  FlatList,
  ActivityIndicator,
  Image,
  TouchableOpacity,
  View,
  TextInput,
  RefreshControl,
} from 'react-native';
import React, { useState, useEffect } from 'react';
import BottomNavigation from '../navigation/BottomNavigation';
import { useNavigation } from '@react-navigation/native';
import MaterialCommunityIcons from 'react-native-vector-icons/MaterialCommunityIcons';
import Feather from 'react-native-vector-icons/Feather';
import AntDesign from 'react-native-vector-icons/AntDesign';
import PostItem from '../components/farmerToFarmer/components/PostItem';
import config, { Config } from 'react-native-config';
import axios from 'axios';
import AsyncStorage from '@react-native-async-storage/async-storage';
import Ionicons from 'react-native-vector-icons/Ionicons';

const FarmerToBuyer = () => {
  const navigation = useNavigation();
  const [searchText, setSearchText] = useState('');
  const [loading, setLoading] = useState(false);


  // search functionality
  const clearSearch = () => {
    setSearchText('');
  };

  const handleSearch = () => {
    setSearchText('');
  };

  return (
    <SafeAreaView style={{ height: '100%', position: 'relative', backgroundColor: '#ffffff' }}>
      {loading ? (
        <View style={styles.loader}>
          <ActivityIndicator size="large" color="#46c67c" />
        </View>
      ) : (
        <View>
          <View style={styles.mainContainer}>
            <View style={styles.topStatusBar}>
              <TouchableOpacity style={styles.backButton} onPress={() => navigation.navigate('FarmerToFarmer')}>
                <Ionicons name="arrow-back" size={22} color="#585C60" />
              </TouchableOpacity>
              <Text style={styles.heading}><Text style={{ fontSize: 20 }}>Agri</Text><Text style={{ color: '#46c67c', fontSize: 20 }}>AI</Text></Text>
            </View>

            <View style={styles.searchContainer}>
              <View style={styles.searchBarLeft}>
                <TextInput
                  style={styles.searchBar}
                  placeholder="Search..."
                  placeholderTextColor={'#605f63'}
                  value={searchText}
                  onChangeText={(text) => setSearchText(text)}
                />
                {searchText ? (
                  <TouchableOpacity onPress={clearSearch} style={styles.cutButton}>
                    <Ionicons name="close" size={20} color="#605f63" />
                  </TouchableOpacity>
                ) : null}
              </View>
              <TouchableOpacity onPress={handleSearch} style={styles.searchButton}>
                <Ionicons name="search" size={20} color="#605f63" />
              </TouchableOpacity>
            </View>

            {/* // category section  */}
            <View style={styles.category}>
              <Text style={styles.categoryHeading}>Category</Text>
              <ScrollView horizontal={true} style={styles.categoryContainer}>
                <TouchableOpacity style={styles.categoryItem}>
                  <View style={styles.ImageContainer}>
                    <Image style={styles.categoryImage} source={require('../assets/icon/fruits.png')} />
                  </View>
                  <Text style={styles.categoryText}>Fruits</Text>
                </TouchableOpacity>
                <TouchableOpacity style={styles.categoryItem}>
                  <View style={styles.ImageContainer}>
                    <Image style={styles.categoryImage} source={require('../assets/icon/vegetable.png')} />
                  </View>
                  <Text style={styles.categoryText}>Vegetable</Text>
                </TouchableOpacity>
                <TouchableOpacity style={styles.categoryItem}>
                  <View style={styles.ImageContainer}>
                    <Image style={styles.categoryImage} source={require('../assets/icon/seed.png')} />
                  </View>
                  <Text style={styles.categoryText}>Seeds</Text>
                </TouchableOpacity>
                <TouchableOpacity style={styles.categoryItem}>
                  <View style={styles.ImageContainer}>
                    <Image style={styles.categoryImage} source={require('../assets/icon/onion.png')} />
                  </View>
                  <Text style={styles.categoryText}>Onion</Text>
                </TouchableOpacity>
                <TouchableOpacity style={styles.categoryItem}>
                  <View style={styles.ImageContainer}>
                    <Image style={styles.categoryImage} source={require('../assets/icon/potato.png')} />
                  </View>
                  <Text style={styles.categoryText}>Potato</Text>
                </TouchableOpacity>
              </ScrollView>
            </View>

            <View style={styles.cropContainer}>
              <Text style={styles.categoryHeading}>Sell Products</Text>
              <View style={styles.cropWrapper}>
                <TouchableOpacity style={styles.cropItem}>
                  <View style={styles.cropImageContainer}>
                    <Image style={styles.cropImage} source={require('../assets/icon/onionSell.jpg')} />
                  </View>
                  <Text style={styles.cropTitle}>Onions 2 Ton</Text>
                  <Text style={styles.cropPrice}>20/kg</Text>
                  <TouchableOpacity style={styles.buyButton}>
                    <Text style={styles.buyButtonText}>Buy</Text>
                  </TouchableOpacity>
                </TouchableOpacity>
                <TouchableOpacity style={styles.cropItem}>
                  <View style={styles.cropImageContainer}>
                    <Image style={styles.cropImage} source={require('../assets/icon/sellPotato.jpg')} />
                  </View>
                  <Text style={styles.cropTitle}>Potato 20 Ton</Text>
                  <Text style={styles.cropPrice}>40/kg</Text>
                  <TouchableOpacity style={styles.buyButton}>
                    <Text style={styles.buyButtonText}>Buy</Text>
                  </TouchableOpacity>
                </TouchableOpacity>
                <TouchableOpacity style={styles.cropItem}>
                  <View style={styles.cropImageContainer}>
                    <Image style={styles.cropImage} source={require('../assets/icon/sellTomato.jpg')} />
                  </View>
                  <Text style={styles.cropTitle}>Tomato 200 Caret</Text>
                  <Text style={styles.cropPrice}>66/kg</Text>
                  <TouchableOpacity style={styles.buyButton}>
                    <Text style={styles.buyButtonText}>Buy</Text>
                  </TouchableOpacity>
                </TouchableOpacity>
              </View>
            </View>
          </View>
        </View>
      )}
      <View style={styles.bottomNav}>
        <BottomNavigation />
      </View>
    </SafeAreaView>
  );
};

const styles = StyleSheet.create({
  mainContainer: {
    height: '100%',
    backgroundColor: '#ffffff',
  },
  loader: {
    borderWidth: 1,
    height: '100%',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
  },
  topStatusBar: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'flex-start',
    flexDirection: 'row',
    gap: 5,
    paddingHorizontal: 10,
    paddingVertical: 10,
    backgroundColor: '#ffffff'
  },
  backButton: {
    // borderWidth:1,
    paddingVertical: 10,
    paddingHorizontal: 10
  },
  heading: {
    color: '#000000',
    fontSize: 25,
    fontWeight: '600'
  },
  searchContainer: {
    marginHorizontal: 20,
    marginBottom: 20,
    marginTop: 5,
    flexDirection: 'row',
    alignItems: 'center',
    borderRadius: 5,
    backgroundColor: '#f8f7fd',
    height: 40
  },
  searchBarLeft: {
    flexDirection: 'row',
    alignItems: 'center',
    width: '86%',
  },
  searchBar: {
    display: 'flex',
    paddingVertical: 5,
    color: '#605f63',
    fontSize: 14,
    paddingHorizontal: 10,
    width: '90%'
  },
  cutButton: {
    height: '100%',
  },
  searchButton: {
    width: '12%',
    height: '100%',
    display: 'flex',
    justifyContent: 'center',
    alignItems: 'center',
  },


  // category
  category: {
    // borderWidth:1,
    marginHorizontal: 20
  },
  categoryHeading: {
    fontSize: 14,
    color: '#3f484b',
    fontWeight: '600',
    marginBottom: 10,
  },
  categoryContainer: {
    gap: 10
  },
  categoryItem: {
    // borderWidth: 1,
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    marginRight: 15,
    borderRadius: 7
  },
  ImageContainer: {
    borderRadius: 7,
    width: 50,
    height: 50,
    alignItems: 'center',
    justifyContent: 'center',
    // borderWidth: 1,
    backgroundColor: '#e9f1f1'
  },
  categoryImage: {
    width: '60%',
    height: '60%'
  },
  categoryText: {
    color: '#3f484b',
    fontSize: 12,
    fontWeight: '500'
  },

  // crop 
  cropContainer: {
    marginHorizontal: 20,
    marginTop: 20
  },
  cropWrapper: {
    flexDirection: 'row',
    justifyContent:'space-between',
    flexWrap:'wrap',
  },
  cropItem: {
    borderWidth: 0.2,
    width: '47%',
    paddingVertical: 13,
    paddingHorizontal: 13,
    borderRadius: 7,
    marginBottom:19
  },
  cropImageContainer: {
    width: '100%',
    height: 100,
    borderRadius: 7
  },
  cropImage: {
    width: '100%',
    height: '100%',
    borderRadius: 7
  },
  cropTitle: {
    color: '#3f484b',
    fontSize: 14,
    fontWeight: '500',
    marginTop: 5
  },
  cropPrice: {
    color: '#46c67c',
    fontSize: 14,
    fontWeight: '600'
  },
  buyButton: {
    backgroundColor: '#46c67c',
    paddingVertical: 7,
    borderRadius: 7,
    alignItems: 'center',
    justifyContent: 'center',
    marginTop: 10
  },
  buyButtonText: {
    fontWeight: '600',
    color: '#ffffff'
  },

  bottomNav: {
    position: 'absolute',
    bottom: 0,
    width: '100%',
  },
});

export default FarmerToBuyer;
