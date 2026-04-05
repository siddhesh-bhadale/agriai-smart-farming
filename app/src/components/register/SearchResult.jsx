{ SafeAreaView, ScrollView, StyleSheet, Text, View } from 'react-native'
import 
  return () => {
    second
  }
}, [third])
} from 'react'
import Ionicons from 'react-native-vector-icons/Ionicons';

const SearchResult = ({ route }) => {
    const { type } = route.params;
    const [isLoading, setIsLoading] = useState(true);

    console.log(type)
    return (
        <SafeAreaView>
            {isLoading ? (
                <ActivityIndicator style={styles.loader} size="large" color="#5525f5" />
            ) : (
                <ScrollView>
                    <View style={styles.mainContainer}>
                        <View style={styles.topStatusBar}>
                            <TouchableOpacity style={styles.backButton} onPress={() => navigation.navigate('FarmerToFarmer')}>
                                <Ionicons name="arrow-back" size={22} color="#585C60" />
                            </TouchableOpacity>
                            <Text style={styles.heading}>Settings</Text>
                        </View>
                    </View>
                </ScrollView>
            )}
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
})

export default SearchResult
