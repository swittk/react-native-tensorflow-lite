import * as React from 'react';

import { StyleSheet, View, Text, Alert, Button, Image, ScrollView, StyleProp, ViewStyle } from 'react-native';
import TensorflowLite, { SKTFLiteTensorResult } from '@switt/react-native-tensorflow-lite';
import { Asset } from 'expo-asset';
import * as ImagePicker from 'expo-image-picker';
import { base64RawToDataURI } from './utils';


export const TestImageScaler = React.memo((props: {
  style?: StyleProp<ViewStyle>,
  imageUri: string,
}) => {
  const {
    style,
    imageUri
  } = props;

  const [transformedImage, setTransformedImage] = React.useState<string>()

  return <View style={style}>
    <Image source={{ uri: imageUri, width: 250, height: 250 }} resizeMode='contain' />
    <Button title='TRANSFORM!' onPress={async () => {
      const result = await TensorflowLite.tensorImageTest({
        file: imageUri,
        // size: { width: 100, height: 100 },
        relativeCrops: [0, 0.0, 0.5, 0.5],
        backgroundColor: '#000000'
      });
      setTransformedImage(base64RawToDataURI(result));
    }} />
    <Text>Transformed Image Result</Text>
    <Image source={{ uri: transformedImage, width: 250, height: 250 }} resizeMode='contain' />
    <Button title='Transformed image info' onPress={async () => {
      if(!transformedImage) {Alert.alert('no image'); return;}
      const prom = new Promise<{width: number, height: number}>((resolve, reject)=>{
        Image.getSize(transformedImage, (w,h)=>{
          resolve({width: w, height: h});
        }, reject);
      });
      const dims = await prom;
      Alert.alert(`Image dims`, JSON.stringify(dims));
    }} />
  </View>
})