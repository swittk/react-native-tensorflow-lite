import { NativeModules } from 'react-native';

export type SKTFLiteRunModelWithFilesArg = {
  /** The path to the tensorflow lite model to use */
  model: string,
  /** The paths to the files to evaluate*/
  files: string[],
  /** The shapes of the imput tesors. CAREFUL THIS WILL CRASH YOUR APP IF NOT PROPERLY DEFINED
   * If this is not provided, the shapes are used from the model anyway.
   */
  shapes?: number[][]
}
export type SKTFLiteTensorResult = SKTFLiteSingleTensorResult[];

export type SKTFLiteSingleTensorResult = {
  shape: number[],
  data: number[]
}

type TensorflowLiteType = {
  multiply(a: number, b: number): Promise<number>;
  /**
   * 
   * @param params 
   * @returns Raw number values of each output tensor, the shapes are not preserved
   */
  runModelWithFiles(params: SKTFLiteRunModelWithFilesArg): Promise<SKTFLiteTensorResult>
};

const { TensorflowLite } = NativeModules;

export default TensorflowLite as TensorflowLiteType;
