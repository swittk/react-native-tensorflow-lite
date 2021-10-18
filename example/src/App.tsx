import * as React from 'react';

import { StyleSheet, View, Text, Alert, Button, Image, ScrollView } from 'react-native';
import TensorflowLite, { SKTFLiteTensorResult } from '@switt/react-native-tensorflow-lite';
import { Asset } from 'expo-asset';
import * as ImagePicker from 'expo-image-picker';
import { TestImageScaler } from './TestImageScaler';


type Props = {};
type State = {
  imageUri?: string,
  results?: SKTFLiteTensorResult[]
}
export default class App extends React.PureComponent<Props, State> {
  constructor(props: Props) {
    super(props);
    this.state = {};
  }
  checkSampleModelParams = async () => {
    // const modelAsset = Asset.fromModule(require('./samplemodel.tflite'));
    const modelAsset = Asset.fromModule(require('./fullfacetwitch.tflite'));
    if (!modelAsset.downloaded) { await modelAsset.downloadAsync() };
    try {
      const results = await TensorflowLite.getModelInfo({
        model: modelAsset.localUri!,
      });
      Alert.alert(`Got Model Info`, JSON.stringify(results));
    } catch (e) {
      Alert.alert(`Model info failed with error ${e.message || e.code} (${JSON.stringify(e)})`)
    }
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

  runGrayscaleModel = async () => {
    const { imageUri } = this.state;
    if (!imageUri) {
      Alert.alert('No Image selected');
      return;
    }
    const modelAsset = Asset.fromModule(require('./fullfacetwitch.tflite'));
    if (!modelAsset.downloaded) { await modelAsset.downloadAsync() };
    try {
      const results = await TensorflowLite.runModelWithFiles({
        model: modelAsset.localUri!,
        shapes: [[432, 432]],
        files: [
          imageUri, imageUri, imageUri, imageUri, imageUri,
          imageUri, imageUri, imageUri, imageUri, imageUri,
          imageUri, imageUri, imageUri, imageUri, imageUri,
          imageUri
        ],
        groupMode: {
          numPerGroup: 16,
        },
        grayscale: true
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
          <Button title="Run grayscale model" onPress={this.runGrayscaleModel} />
          <Button title="try check params" onPress={this.checkSampleModelParams} />
          <Image source={{ uri: imageUri }} style={{ width: 320, height: 320 }} resizeMode='contain' />
          <Text>Result: {JSON.stringify(results)}</Text>
          {imageUri && <TestImageScaler
            imageUri={imageUri}
            faceLandmarkResults={results}
          />}
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
