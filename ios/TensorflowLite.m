#import "TensorflowLite.h"
#import "TFLTensorFlowLite.h"
#import <UIKit/UIKit.h>
#import "TFRelatedExtensions.h"
//#import "UIImage-Swift.h"

@implementation TensorflowLite {
    // Potentially allow for multiple interpreters
    TFLInterpreter *interp1;
    TFLInterpreter *interp2;
    TFLInterpreter *interp3;
}

RCT_EXPORT_MODULE()

// Example method
// See // https://reactnative.dev/docs/native-modules-ios
RCT_REMAP_METHOD(multiply,
                 multiplyWithA:(nonnull NSNumber*)a withB:(nonnull NSNumber*)b
                 withResolver:(RCTPromiseResolveBlock)resolve
                 withRejecter:(RCTPromiseRejectBlock)reject)
{
    NSNumber *result = @([a floatValue] * [b floatValue]);
    
    resolve(result);
}

RCT_REMAP_METHOD(getModelInfo,
                 getModelInfo:(nonnull NSDictionary *)argumentsDict
                 withResolver:(RCTPromiseResolveBlock)resolve
                 withRejecter:(RCTPromiseRejectBlock)reject)
{
    NSError *error;
    NSString *modelPath = argumentsDict[@"model"];
    if(!modelPath) {
        error = [[NSError alloc] initWithDomain:@"SKRNTFLITE" code:1 userInfo:nil];
        reject(@"NO_MODEL_PATH", @"No Model Path specified", error);
        return;
    }
    if([modelPath hasPrefix:@"file://"]) {
        modelPath = [modelPath substringFromIndex:7];
    }
    
    TFLInterpreter *interpreter = [[TFLInterpreter alloc] initWithModelPath:modelPath
                                                                      error:&error];
    if (error != nil) { /* Error handling... */
        reject(@"TF_LOAD_ERROR", @"TF Lite model load failed", error);
        return;
    }
    [interpreter allocateTensorsWithError:&error];
    if (error != nil) { /* Error handling... */
        reject(@"TF_INIT_ALLOC_ERROR", @"TF Lite init allocation failed", error);
        return;
    }
    NSUInteger tensorCount = [interpreter inputTensorCount];
    NSUInteger outTensorCount = [interpreter outputTensorCount];
    if(tensorCount == 0 || outTensorCount == 0) {
        error = [[NSError alloc] initWithDomain:@"SKRNTFLITE" code:30 userInfo:nil];
        reject(@"TF_TENSOR_ALLOC_ERROR", @"TF Lite interpreter creation failed", error);
        return;
    }
    /**
     * Input shapes dictionary array
     * Each element has { name, dataType, shape }
     */
    NSMutableArray <NSDictionary *>*inputShapes = [NSMutableArray new];
    for(NSUInteger i = 0; i < tensorCount; i++) {
        TFLTensor *tensor = [interpreter inputTensorAtIndex:i error:&error];
        NSArray <NSNumber *>*tensorshape = [tensor shapeWithError:&error];
        if(error) {
            reject(
                   @"TF_SHAPE_ERROR",
                   [NSString stringWithFormat:@"TF Lite failed to get shape for input tensor at index %lu", i],
                   error);
            return;
        }
        [inputShapes addObject:@{ @"name": tensor.name ?:@"NO_NAME", @"dataType": @(tensor.dataType), @"shape": tensorshape }];
    }
    
    NSMutableArray <NSDictionary *>*outputShapes = [NSMutableArray new];
    for(NSUInteger i = 0; i < outTensorCount; i++) {
        TFLTensor *tensor = [interpreter outputTensorAtIndex:i error:&error];
        NSArray <NSNumber *>*tensorshape = [tensor shapeWithError:&error];
        if(error) {
            reject(
                   @"TF_SHAPE_ERROR",
                   [NSString stringWithFormat:@"TF Lite failed to get shape for output tensor at index %lu", i],
                   error);
            return;
        }
        [outputShapes addObject:@{ @"name": tensor.name ?:@"NO_NAME", @"dataType": @(tensor.dataType), @"shape": tensorshape }];
    }
    resolve(@{
        @"input": inputShapes,
        @"output": outputShapes
            });
}

RCT_REMAP_METHOD(runModelWithFiles,
                 runModelWithFiles:(nonnull NSDictionary *)argumentsDict
                 withResolver:(RCTPromiseResolveBlock)resolve
                 withRejecter:(RCTPromiseRejectBlock)reject)
{
    NSError *error;
    NSString *modelPath = argumentsDict[@"model"];
    NSArray <NSString *>*filePaths = argumentsDict[@"files"];
    NSString *fileMode = argumentsDict[@"fileMode"];
    // User-provided shapes for tensors (Optional, can be inferred from the model)
    NSArray <NSArray <NSNumber *>*>*shapes = argumentsDict[@"shapes"];
    // User-provided groupMode configuration options.
    // If specified, the files are inputted to the model in `group`s
    NSDictionary *groupMode = argumentsDict[@"groupMode"];
    BOOL imageModeIsFit = [argumentsDict[@"imageScaleMode"] isEqualToString:@"fit"];
    NSArray <NSArray <NSNumber *>*>*imageCrops = argumentsDict[@"imageCrops"];
    
    int stride = 1;
    int numPerGroup = 1;
    if(groupMode) {
        stride = [groupMode[@"stride"] intValue] ?: 1;
        numPerGroup = [groupMode[@"numPerGroup"] intValue] ?: 1;
        if(stride <= 0) {
            error = [[NSError alloc] initWithDomain:@"SKRNTFLITE" code:94 userInfo:nil];
            reject(@"INVALID_PARAMETER", @"stride must be more than zero", error);
            return;
        }
        if(numPerGroup <= 0) {
            error = [[NSError alloc] initWithDomain:@"SKRNTFLITE" code:94 userInfo:nil];
            reject(@"INVALID_PARAMETER", @"numPerGroup must be more than zero", error);
            return;
        }
    }
    
    if((fileMode != nil) && ![fileMode isEqualToString:@"image"]) {
        error = [[NSError alloc] initWithDomain:@"SKRNTFLITE" code:45 userInfo:nil];
        reject(@"UNSUPPORTED_MODE", @"Only image mode is currently supported", error);
        return;
        // TODO: ENABLE FILE MODE FOR BOTH IMAGE AND REGULAR FILE
    }
    if(!modelPath) {
        error = [[NSError alloc] initWithDomain:@"SKRNTFLITE" code:1 userInfo:nil];
        reject(@"NO_MODEL_PATH", @"No Model Path specified", error);
        return;
    }
    else if(!filePaths) {
        error = [[NSError alloc] initWithDomain:@"SKRNTFLITE" code:1 userInfo:nil];
        reject(@"NO_FILE_PATHS", @"No file paths specified", error);
        return;
    }
    if([modelPath hasPrefix:@"file://"]) {
        modelPath = [modelPath substringFromIndex:7];
    }
    
    TFLInterpreter *interpreter = [[TFLInterpreter alloc] initWithModelPath:modelPath
                                                                      error:&error];
    if (error != nil) { /* Error handling... */
        reject(@"TF_LOAD_ERROR", @"TF Lite model load failed", error);
        return;
    }
    [interpreter allocateTensorsWithError:&error];
    if (error != nil) { /* Error handling... */
        reject(@"TF_INIT_ALLOC_ERROR", @"TF Lite init allocation failed", error);
        return;
    }
    
    NSUInteger tensorCount = [interpreter inputTensorCount];
    NSUInteger outTensorCount = [interpreter outputTensorCount];
    if(tensorCount == 0 || outTensorCount == 0) {
        error = [[NSError alloc] initWithDomain:@"SKRNTFLITE" code:30 userInfo:nil];
        reject(@"TF_TENSOR_ALLOC_ERROR", @"TF Lite interpreter creation failed", error);
        return;
    }
    else if(tensorCount > 1) {
        error = [[NSError alloc] initWithDomain:@"SKRNTFLITE" code:30 userInfo:nil];
        reject(@"TF_TENSOR_ALLOC_ERROR", [NSString stringWithFormat:@"Tensor count is %d, only support 1 tensor right now", tensorCount], error);
        return;
    }
    /**
     BatchOutput:: Each element = each input file
     element keys are "length", "data", "shape"
     ( length = number of tensors, data = Record<number, number[]>, shape = number[] )
     */
    NSMutableArray *batchOutput = [NSMutableArray new];
    
    // The cap file index to not reach when enumerating the files (limited by group inputs)
    NSInteger capFileIndex = filePaths.count - (numPerGroup - 1);
    for(NSInteger inputIndex = 0; inputIndex < capFileIndex; inputIndex += stride) {
        for(NSUInteger i = 0; i < tensorCount; i++) {
            TFLTensor *tensor = [interpreter inputTensorAtIndex:0 error:&error];
            if(error != nil) {
                reject(
                       @"TF_INPUT_TENSOR_ERROR",
                       [NSString stringWithFormat:@"TF Lite failed to get tensor at index %lu", i],
                       error);
                return;
            }
            CGSize shape;
            if(shapes) {
                shape = CGSizeMake(shapes[i][0].integerValue, shapes[i][1].integerValue);
            }
            else {
                NSArray <NSNumber *>*tensorshape = [tensor shapeWithError:&error];
                if(error) {
                    reject(
                           @"TF_SHAPE_ERROR",
                           [NSString stringWithFormat:@"TF Lite failed to get shape for tensor at index %lu", i],
                           error);
                    return;
                }
                // TODO: MAKE THIS SMARTER NOT THIS DUMB
                if(tensorshape.count > 3) {
                    shape.width = tensorshape[1].intValue;
                    shape.height = tensorshape[2].intValue;
                }
                else {
                    shape.width = tensorshape[0].intValue;
                    shape.height = tensorshape[1].intValue;
                }
            }
            NSData *data;
            if(numPerGroup > 1) {
                NSMutableData *groupData = [NSMutableData new];
                if(imageCrops) {
                    for(int groupIndex = 0; groupIndex < numPerGroup; groupIndex++) {
                        NSInteger inputFileIndex = inputIndex + groupIndex;
                        NSString *filePath = filePaths[inputFileIndex];
                        data = [self tensorImageDataForFilePath:filePath size:shape isFit:imageModeIsFit cropTo:imageCrops[inputFileIndex]];
                        [groupData appendData:data];
                    }
                }
                else {
                    for(int groupIndex = 0; groupIndex < numPerGroup; groupIndex++) {
                        NSInteger inputFileIndex = inputIndex + groupIndex;
                        NSString *filePath = filePaths[inputFileIndex];
                        data = [self tensorImageDataForFilePath:filePath size:shape isFit:imageModeIsFit];
                        [groupData appendData:data];
                    }
                }
                data = groupData;
            }
            else {
                NSString *filePath = filePaths[inputIndex];
                data = [self tensorImageDataForFilePath:filePath size:shape isFit:imageModeIsFit cropTo:imageCrops.count ? imageCrops[0] : nil];
            }
            [tensor copyData:data error:&error];
            if(error != nil) {
                reject(
                       @"TF_INPUT_COPY_ERROR",
                       [NSString stringWithFormat:@"TF Lite failed to copy tensor at index %lu", i],
                       error);
                return;
            }
        }
        BOOL ok = [interpreter invokeWithError:&error];
        if(!ok || (error != nil)) {
            reject(
                   @"TF_INVOKE_ERROR",
                   [NSString stringWithFormat:@"TF Lite invocation failed for file at path %@", filePaths[inputIndex]],
                   error);
            return;
        }
        
        // See documentation on data types at source here
        // https://github.com/tensorflow/tensorflow/blob/b0baa1cbeeb62fc55a21c1ebf980d22e1099fd56/tensorflow/lite/objc/apis/TFLTensor.h
        NSMutableArray *outTensors = [NSMutableArray new];
        // With this implementation it doesn't crash with only one image supplied however..
        // now this is weird, if we call model only once then it does not crash, call it five times however, and it crashes again at the React native bridge
        // Maybe it's something to do with sizes of values that are passed through the bridge then.
        for(NSUInteger i = 0; i < outTensorCount; i++) {
            NSMutableArray *outData = [NSMutableArray new];
            TFLTensor *outputi = [interpreter outputTensorAtIndex:i error:&error];
            NSArray <NSNumber *>*shape = [outputi shapeWithError:&error];
            if(error) {
                reject(@"TF_OUTPUT_ERROR", [NSString stringWithFormat:@"Failed to get shape for tensor at index %lu", i], error);
                return;
            }
            NSData *data = [outputi dataWithError:&error];
            if(error) {
                reject(@"TF_OUTPUT_ERROR", [NSString stringWithFormat:@"Failed to get output data for tensor at index %lu", i], error);
                return;
            }
            switch(outputi.dataType) {
                case TFLTensorDataTypeFloat32: {
                    for(NSUInteger i = 0; i < [data length]; i+=4) {
                        Float32 val;
                        [data getBytes:&val range:NSMakeRange(i, 4)];
                        [outData addObject:@(val)];
                    }
                } break;
                case TFLTensorDataTypeFloat16: {
                    for(NSUInteger i = 0; i < [data length]; i+=2) {
                        Float32 val;
                        [data getBytes:&val range:NSMakeRange(i, 2)];
                        [outData addObject:@(val)];
                    }
                } break;
                case TFLTensorDataTypeInt32: {
                    for(NSUInteger i = 0; i < [data length]; i+=4) {
                        SInt32 val;
                        [data getBytes:&val range:NSMakeRange(i, 4)];
                        [outData addObject:@(val)];
                    }
                } break;
                case TFLTensorDataTypeUInt8: {
                    for(NSUInteger i = 0; i < [data length]; i+=1) {
                        UInt8 val;
                        [data getBytes:&val range:NSMakeRange(i, 1)];
                        [outData addObject:@(val)];
                    }
                } break;
                case TFLTensorDataTypeInt64: {
                    for(NSUInteger i = 0; i < [data length]; i+=8) {
                        SInt64 val;
                        [data getBytes:&val range:NSMakeRange(i, 8)];
                        [outData addObject:@(val)];
                    }
                } break;
                case TFLTensorDataTypeBool: {
                    for(NSUInteger i = 0; i < [data length]; i+=1) {
                        BOOL val;
                        [data getBytes:&val range:NSMakeRange(i, 1)];
                        [outData addObject:@(val)];
                    }
                } break;
                case TFLTensorDataTypeInt16: {
                    for(NSUInteger i = 0; i < [data length]; i+=2) {
                        SInt16 val;
                        [data getBytes:&val range:NSMakeRange(i, 2)];
                        [outData addObject:@(val)];
                    }
                } break;
                case TFLTensorDataTypeInt8: {
                    for(NSUInteger i = 0; i < [data length]; i+=1) {
                        SInt8 val;
                        [data getBytes:&val range:NSMakeRange(i, 1)];
                        [outData addObject:@(val)];
                    }
                } break;
                case TFLTensorDataTypeFloat64: {
                    for(NSUInteger i = 0; i < [data length]; i+=8) {
                        Float64 val;
                        [data getBytes:&val range:NSMakeRange(i, 8)];
                        [outData addObject:@(val)];
                    }
                } break;
                default: {
                    reject(@"TF_OUTPUT_ERROR", @"TFLite encountered an unknown output data type", [NSError new]);
                    return;
                } break;
            }
            [outTensors addObject:@{
                @"shape": shape,
                @"data": outData
            }];
        }
        [batchOutput addObject:outTensors];
    }
    resolve(batchOutput);
}

-(NSData *)tensorImageDataForFilePath:(NSString *)filePath size:(CGSize)shape isFit:(BOOL)imageModeIsFit {
    NSString *actualFilePath;
    if([filePath hasPrefix:@"file://"]) {
        actualFilePath = [filePath substringFromIndex:7];
    }
    else {
        actualFilePath = filePath;
    }
    UIImage *image = [UIImage imageWithContentsOfFile:actualFilePath];
    if(!image) {
        NSLog(@"No image found at given path");
        return nil;
    }
    NSData *data;
    if(imageModeIsFit) {
        data = [[image imageFittedToSize:shape] scaledDataWithSize:shape isQuantized:NO];
    }
    else {
        data = [image scaledDataWithSize:shape isQuantized:NO];
    }
    return data;
}

-(NSData *)tensorImageDataForFilePath:(NSString *)filePath size:(CGSize)shape isFit:(BOOL)imageModeIsFit cropTo:(NSArray <NSNumber *>*)crop {
    NSString *actualFilePath;
    if([filePath hasPrefix:@"file://"]) {
        actualFilePath = [filePath substringFromIndex:7];
    }
    else {
        actualFilePath = filePath;
    }
    UIImage *image = [UIImage imageWithContentsOfFile:actualFilePath];
    if(!image) {
        NSLog(@"No image found at given path");
        return nil;
    }
    if(crop) {
        if([crop count] < 4) {
            NSLog(@"Crop count < 4");
            return nil;
        }
        image = [image cropToX:crop[0].intValue y:crop[1].intValue width:crop[2].intValue height:crop[3].intValue];
    }
    NSData *data;
    if(imageModeIsFit) {
        data = [[image imageFittedToSize:shape] scaledDataWithSize:shape isQuantized:NO];
    }
    else {
        data = [image scaledDataWithSize:shape isQuantized:NO];
    }
    return data;
}


@end
