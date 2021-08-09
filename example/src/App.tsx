import * as React from 'react';

import { StyleSheet, View, Text, Alert, Button, Image, ScrollView } from 'react-native';
import TensorflowLite, { SKTFLiteTensorResult } from 'react-native-tensorflow-lite';
import { Asset } from 'expo-asset';
import * as ImagePicker from 'expo-image-picker';


type Props = {};
type State = {
  imageUri?: string,
  results?: SKTFLiteTensorResult
}
export default class App extends React.PureComponent<Props, State> {
  constructor(props: Props) {
    super(props);
    this.state = {};
  }
  runModel = async () => {
    const { imageUri } = this.state;
    if (!imageUri) {
      Alert.alert('No Image selected');
      return;
    }
    const modelAsset = Asset.fromModule(require('./face_landmark.tflite'));
    if (!modelAsset.downloaded) { await modelAsset.downloadAsync() };
    try {
      const results = await TensorflowLite.runModelWithFiles({
        model: modelAsset.localUri!,
        files: [imageUri]
      });
      this.setState({ results });
    } catch (e) {
      Alert.alert(`Run failed with error ${e.message || e.code} (${JSON.stringify(e)})`)
    }
  }
  runModelFiveTimes = async () => {
    const { imageUri } = this.state;
    if (!imageUri) {
      Alert.alert('No Image selected');
      return;
    }
    const modelAsset = Asset.fromModule(require('./face_landmark.tflite'));
    if (!modelAsset.downloaded) { await modelAsset.downloadAsync() };
    try {
      const results = await TensorflowLite.runModelWithFiles({
        model: modelAsset.localUri!,
        files: [imageUri, imageUri, imageUri, imageUri, imageUri]
      });
      this.setState({ results });
    } catch (e) {
      Alert.alert(`Run failed with error ${e.message || e.code} (${JSON.stringify(e)})`)
    }
  }

  pickImage = async () => {
    let { status, canAskAgain } = await ImagePicker.getMediaLibraryPermissionsAsync();
    if (status != 'granted') {
      if (!canAskAgain) {
        Alert.alert('Cannot pick, permission was denied, please open settings');
        return;
      }
      ({ status } = await ImagePicker.requestMediaLibraryPermissionsAsync());
    }
    if (status != 'granted') {
      Alert.alert('Cannot browse, no permission');
      return;
    }
    const result = await ImagePicker.launchImageLibraryAsync({ mediaTypes: ImagePicker.MediaTypeOptions.Images });
    if (result.cancelled) {
      return;
    }
    const uri = result.uri;
    this.setState({ imageUri: uri });
  }

  render() {
    const { results, imageUri } = this.state;
    // const [result, setResult] = React.useState<number | undefined>();

    // React.useEffect(() => {
    //   TensorflowLite.multiply(3, 7).then(setResult);
    // }, []);

    return (
      <View style={styles.container}>
        <ScrollView style={{ flex: 1 }}>
          <Button title="Pick image" onPress={this.pickImage} />
          <Button title="Run model" onPress={this.runModel} />
          <Button title="Run model 5 times" onPress={this.runModelFiveTimes} />
          <Image source={{ uri: imageUri }} style={{ width: 320, height: 320 }} resizeMode='contain' />
          <Text>Result: {JSON.stringify(results)}</Text>
        </ScrollView>
      </View>
    );
  }
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
  },
  box: {
    width: 60,
    height: 60,
    marginVertical: 20,
  },
});
