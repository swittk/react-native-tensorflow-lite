import { NativeModules } from 'react-native';

type TensorflowLiteType = {
  multiply(a: number, b: number): Promise<number>;
};

const { TensorflowLite } = NativeModules;

export default TensorflowLite as TensorflowLiteType;
