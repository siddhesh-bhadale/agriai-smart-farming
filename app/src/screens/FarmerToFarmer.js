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

const FarmerToFarmer = () => {
  const navigation = useNavigation();
  const [search, setSearch] = useState('');
  const [loading, setLoading] = useState(true);
  const [posts, setPosts] = useState([]);
  const [userData, setUserData] = useState(null);
  const [refreshing, setRefreshing] = useState(false);

  useEffect(() => {
    const loadUserData = async () => {
      try {
        const storedUserData = await AsyncStorage.getItem('userData');
        setUserData(JSON.parse(storedUserData));
      } catch (error) {
        console.error('Error loading userData:', error);
      }
    };

    loadUserData();
  }, []);

  useEffect(() => {
    const fetchPosts = async () => {
      setLoading(true);
      try {
        if (userData) {
          const response = await axios.get(`${Config.API_URL}/posts?userId=${userData._id}`);
          setPosts(response.data);
        }
      } catch (err) {
        console.log(err.message);
      } finally {
        setLoading(false);
      }
    };

    fetchPosts();
  }, [userData]);

  const onRefresh = async () => {
    setRefreshing(true);
    try {
      if (userData) {
        const response = await axios.get(`${Config.API_URL}/posts?userId=${userData._id}`);
        setPosts(response.data);
      }
    } catch (err) {
      console.log(err.message);
    } finally {
      setRefreshing(false);
    }
  };

  const renderItem = ({ item }) => <PostItem item={item} />;

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
              <TouchableOpacity style={styles.profileImageContainer} onPress={() => navigation.navigate('Profile')}>
                <Image
                  style={styles.profileImage}
                  source={
                    userData && userData.image
                      ? { uri: `${Config.API_URL}${userData.image}` }
                      : require('../assets/icon/user.jpg')
                  }
                />
              </TouchableOpacity>
              <View style={styles.searchBarContainer}>
                <View style={styles.searchIcon}>
                  <Feather name="search" size={22} color="#585C60" />
                </View>
                <TextInput
                  style={styles.search}
                  placeholder="Search..."
                  placeholderTextColor={"#828282"}
                  value={search}
                  autoCapitalize="none"
                  autoCorrect={false}
                  onChangeText={setSearch}
                />
              </View>
              <TouchableOpacity style={styles.messageContainer} onPress={() => navigation.navigate('Message')}>
                <MaterialCommunityIcons name="message-reply-text" size={26} color="#585C60" />
              </TouchableOpacity>
            </View>

            <View style={styles.postContainer}>
              {posts.length > 0 ? (
                <FlatList
                  data={posts}
                  keyExtractor={(item) => item._id.toString()}
                  renderItem={renderItem}
                  refreshControl={
                    <RefreshControl refreshing={refreshing} onRefresh={onRefresh} />
                  }
                />
              ) : (
                <Text style={{ color: '#000000', textAlign: 'center' }}>No any Post uploaded!</Text>
              )}
            </View>
          </View>
          <TouchableOpacity style={styles.addFarmerButton} onPress={() => navigation.navigate('AddFarmer')}>
            <AntDesign name="adduser" size={24} color="#ffffff" />
          </TouchableOpacity>
          <View syle={{ height: 300, paddingHorizontal: 300 }}></View>
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
    justifyContent: 'space-between',
    flexDirection: 'row',
    width: '100%',
    paddingHorizontal: 20,
    paddingVertical: 20,
  },
  profileImageContainer: {
    height: 35,
    width: 35,
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    borderRadius: 100,
    overflow: 'hidden',
  },
  profileImage: {
    width: '100%',
    height: '100%',
  },
  searchBarContainer: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'flex-start',
    flexDirection: 'row',
    backgroundColor: '#eeeeee',
    height: 35,
    width: 220,
    borderRadius: 5,
  },
  searchIcon: {
    paddingHorizontal: 5,
  },
  search: {
    backgroundColor: '#eeeeee',
    color: '#000000',
    height: 35,
    paddingVertical: -5,
    width: 185,
    borderRadius: 5,
  },

  // post styling
  postContainer: {
    paddingHorizontal: 20,
    marginBottom: 135,
  },

  // add farmer
  addFarmerButton: {
    backgroundColor: '#46c67c',
    width: 50,
    height: 50,
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    borderRadius: 100,
    position: 'absolute',
    right: 15,
    bottom: 80,
  },
  block: {
    width: '100%',
    height: 300,
    borderWidth: 1,
  },
  bottomNav: {
    position: 'absolute',
    bottom: 0,
    width: '100%',
  },
});

export default FarmerToFarmer;
