import { ScrollView, SafeAreaView, StyleSheet, Text, Image, View, Dimensions, TouchableOpacity, ActivityIndicator } from 'react-native'
import React, { useState, useEffect } from 'react'
import Tts from 'react-native-tts';
import Feather from 'react-native-vector-icons/Feather'

const DiseaseDetails = ({ route }) => {
    const [jsonData, setJsonData] = useState(null);
    const [textToSpeak, setTextToSpeak] = useState('');
    const [isSpeaking, setIsSpeaking] = useState(false);
    const [loading, setLoading] = useState(false);

    useEffect(() => {
        Tts.setDefaultLanguage('en-IN');
    }, []);

    const handleSpeak = () => {
        if (isSpeaking) {
            Tts.stop();
        } else {
            Tts.speak(textToSpeak);
        }

        setIsSpeaking(!isSpeaking);
    };


    useEffect(() => {
        // Load the JSON file
        const loadData = async () => {
            setLoading(true)
            try {
                const inputDiseaseName = route.params.result.name === 'Healthy' ? route.params.result.type + ' ' + route.params.result.name : route.params.result.name;

                const json = require('../../data/cropDiseaseDetails.json');

                const filteredData = json.filter((item) => {
                    // Case-insensitive comparison
                    return item.diseaseName.toLowerCase() === inputDiseaseName.toLowerCase();
                });

                setJsonData(filteredData);

                if (route.params.result.name === 'Healthy') {
                    const text = `it is ${route.params.result.type} crop and is ${filteredData[0].diseaseName}. Growing environment : ${filteredData[0].symptoms}.`
                    setTextToSpeak(text)
                } else {
                    const text = `it is ${route.params.result.type} crop and disease name is ${filteredData[0].diseaseName}. symptoms : ${filteredData[0].symptoms}. cause : ${filteredData[0].cause}. comments : ${filteredData[0].comments}. management : ${filteredData[0].management} `
                    setTextToSpeak(text)
                }
                setLoading(false)
            } catch (error) {
                console.error('Error loading JSON:', error);
                setLoading(false)
            }
        };

        const filterDataByDiseaseName = (inputDiseaseName) => {
            const filteredData = jsonData.filter((item) => {
                // Case-insensitive comparison
                return item.diseaseName.toLowerCase() === inputDiseaseName.toLowerCase();
            });

            return filteredData;
        };

        // Call the async function
        loadData();
    }, []);




    return (
        <SafeAreaView>
            {jsonData ? (
                <ScrollView>
                    <View style={styles.container}>
                        <View style={styles.imageContainer}>
                            <Image source={{ uri: jsonData[0].image }} style={styles.detailImage} />
                        </View>
                        <Text style={styles.headingText}>Details</Text>
                        <View style={styles.type}>
                            <View>
                                <Text style={styles.typeText}>{route.params.result.type}</Text>
                                <Text style={styles.diseaseName}>{jsonData[0].diseaseName}</Text>
                            </View>
                            {isSpeaking ? (
                                <TouchableOpacity style={styles.speakButton} onPress={handleSpeak}>
                                    <Feather name="pause-circle" size={34} color="#46c67c" />
                                </TouchableOpacity>
                            ) : (
                                <TouchableOpacity style={styles.speakButton} onPress={handleSpeak}>
                                    <Feather name="play-circle" size={34} color="#46c67c" />
                                </TouchableOpacity>
                            )}

                        </View>

                        {route.params.result.name === 'Healthy' ? (
                            <View style={styles.details}>
                                <View style={styles.textBox}>
                                    <Text style={styles.detailsHeading}>Growing environment :</Text>
                                    <Text style={styles.detailsPara}>{jsonData[0].symptoms}</Text>
                                </View>
                            </View>
                        ) : (
                            <View style={styles.details}>
                                <View style={styles.textBox}>
                                    <Text style={styles.detailsHeading}>symptoms :</Text>
                                    <Text style={styles.detailsPara}>{jsonData[0].symptoms}</Text>
                                </View>
                                <View style={[styles.textBox, { flexDirection: 'row' }]}>
                                    <Text style={styles.detailsHeading}>cause :</Text>
                                    <Text style={styles.detailsPara}>{jsonData[0].cause}</Text>
                                </View>
                                <View style={styles.textBox}>
                                    <Text style={styles.detailsHeading}>comments :</Text>
                                    <Text style={styles.detailsPara}>{jsonData[0].comments}</Text>
                                </View>
                                <View style={styles.textBox}>
                                    <Text style={styles.detailsHeading}>management :</Text>
                                    <Text style={styles.detailsPara}>{jsonData[0].management}</Text>
                                </View>
                            </View>
                        )}

                    </View>
                </ScrollView>
            ) : (
                <View style={styles.loader}>
                    <ActivityIndicator size="large" color="#054D3B" />
                    <Text style={{ color: '#000', marginTop: 5 }}>Loading...</Text>
                </View>
            )}
        </SafeAreaView>
    )
}

const styles = StyleSheet.create({
    loader: {
        borderWidth: 1,
        height: '100%',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
    },
    imageContainer: {
        width: Dimensions.get('window').width,
        height: 200,
        borderBottomLeftRadius: 30,
        borderBottomRightRadius: 30,
        overflow: 'hidden'
    },
    detailImage: {
        height: '100%',
        width: '100%',
        resizeMode: 'cover',
    },
    headingText: {
        color: '#ffffff',
        position: 'absolute',
        top: 30,
        zIndex: 3,
        textAlign: 'center',
        width: '100%',
        fontSize: 20,
        fontWeight: '600'
    },
    type: {
        marginTop: 15,
        flexDirection: 'row',
        alignItems: 'center',
        justifyContent: 'space-between',
        marginHorizontal: 15
    },
    typeText: {
        fontSize: 17,
        fontWeight: '600',
        color: '#000000',
    },
    diseaseName: {
        fontSize: 14,
        fontWeight: '400',
        color: '#000000',
    },
    speakButton: {
        width: 35,
        height: 35
    },
    speakIcon: {
        width: '100%',
        height: '100%'
    },
    details: {
        paddingHorizontal: 15,
        marginTop: 10
    },
    textBox: {
        marginTop: 10
    },
    detailsHeading: {
        color: '#000000',
        fontSize: 15,
        fontWeight: '600',
        textTransform: 'capitalize',
        marginBottom: 3
    },
    detailsPara: {
        color: '#000000',
        fontSize: 13,
        fontWeight: '300',
        textTransform: 'capitalize'
    },
})
export default DiseaseDetails

