import { ActivityIndicator, SafeAreaView, ScrollView, StyleSheet, Text, TextInput, TouchableOpacity, View } from 'react-native'
import React, { useEffect, useRef, useState } from 'react'
import { useNavigation } from '@react-navigation/native';
import Ionicons from 'react-native-vector-icons/Ionicons';
import AsyncStorage from '@react-native-async-storage/async-storage';
import Config from 'react-native-config';
import axios from 'axios';
import UserChat from './components/UserChat';
import BottomNavigation from '../../navigation/BottomNavigation';
import { io } from "socket.io-client";

const Message = () => {
  const navigation = useNavigation();
  const socket = useRef();
  const [isLoading, setIsLoading] = useState(false);
  const [searchText, setSearchText] = useState('');
  const [acceptedFriends, setAcceptedFriends] = useState([]);
  const [userId, setUserId] = useState('');


  useEffect(() => {
    const acceptedFriendsList = async () => {
      try {
        setIsLoading(true)

        // get user id
        const storedUserData = await AsyncStorage.getItem('userData');
        const userData = JSON.parse(storedUserData);
        setUserId(userData._id);

        const response = await fetch(
          `${Config.API_URL}/accepted-friends/${userData._id}`
        );
        const data = await response.json();

        if (response.ok) {
          setAcceptedFriends(data);
          setIsLoading(false);
        }
      } catch (error) {
        console.log("error showing the accepted friends", error);
        setIsLoading(false);
      }
    };

    acceptedFriendsList();
  }, []);
  // console.log("friends", acceptedFriends)


  useEffect(() => {
    if (userId) {
      socket.current = io(Config.API_URL);
      socket.current.emit("add-user", userId);
    }
  }, [userId]);


  //search functinality
  const clearSearch = () => {
    setSearchText('');
  };

  const handleSearch = () => {
    setSearchText('');
  };
  return (
    <SafeAreaView style={{ height: '100%', backgroundColor: '#ffffff' }}>
      {isLoading ? (
        <ActivityIndicator style={styles.loader} size="large" color="#46c67c" />
      ) : (
        <ScrollView>
          <View style={styles.mainContainer}>
            <View style={styles.topStatusBar}>
              <TouchableOpacity style={styles.backButton} onPress={() => navigation.goBack()}>
                <Ionicons name="arrow-back" size={22} color="#585C60" />
              </TouchableOpacity>
              <Text style={styles.heading}>Chats</Text>
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
              < TouchableOpacity onPress={handleSearch} style={styles.searchButton}>
                <Ionicons name="search" size={20} color="#605f63" />
              </TouchableOpacity>
            </View>

            {/* chats  */}
            <View style={styles.chats}>
              {acceptedFriends.map((item, index) => (
                <UserChat key={index} item={item} userId={userId} socket={socket} />
              ))}
            </View>

          </View>
        </ScrollView>
      )}
      <BottomNavigation />
    </SafeAreaView>
  )
}

const styles = StyleSheet.create({
  mainContainer: {
    backgroundColor: '#ffffff',
    height: '100%'
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
    fontSize: 16,
    fontWeight: '600'
  },

  // search 
  // serach 
  searchContainer: {
    marginHorizontal: 20,
    marginBottom: 20,
    marginTop: 5,
    flexDirection: 'row',
    alignItems: 'center',
    borderRadius: 5,
    // borderWidth: 1,
    // borderColor: '#cccccc',
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

  // chats 
  chats: {
    marginHorizontal: 20,
    borderBottomWidth: 0.2,
    borderBottomColor: '#e0e0e0',
    borderTopWidth: 0.2,
    borderTopColor: '#e0e0e0',
  }

})

export default Message
