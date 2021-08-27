import torch
import torch.nn as nn
from torch.nn.modules import batchnorm, dropout
import torch.utils.data as data 

import torchvision
from torchvision import transforms

from PIL import Image
import numpy as np
import os
import yaml
import math
import json

class Dataset (data.Dataset) :
    def __init__ (self, dataset_props, filename) :
        super (Dataset, self).__init__ ()
        self.dataset_props = dataset_props
        self.filename = filename

        self.image_width = self.dataset_props.get ("image_width", 256)
        self.image_height = self.dataset_props.get ("image_height", 256)
        self.transform = transforms.Compose ([
            transforms.Resize ((self.image_height, self.image_width)),
            transforms.ToTensor (),
            transforms.Normalize (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.data = [] 

        file = open (filename)
        for line in file.readlines ()[1:] :
            if filename.split('.')[-1] == 'csv' :
                self.data.append (line.split(','))
            elif filename.split('.')[-1] == 'txt' :
                self.data.append (line.split(' '))
            else :
                raise Exception ("datafile format not known. please use one of .csv or .txt")
    
    def __getitem__ (self, index) :
        datapoint = self.data[index]
        if len (datapoint) != 2 :
            raise Exception ("make sure training and testing files are in correct format")

        image_path = datapoint[0]
        try :
            label = int (datapoint[1])
        except Exception as e :
            raise Exception ("make sure training and testing files are in correct format")

        image = Image.open (image_path).convert ('RGB')
        image = self.transform (image)

        return image, label

    def __len__ (self) :
        return len (self.data)

class Model (nn.Module) :
    def __init__ (self, model_arguments) :
        super ().__init__ ()
        self.model_arguments = model_arguments

        default_channels = 3

        self.layers = []
        conv_layers = model_arguments.get ('conv_layers')
        for layer in conv_layers :
            in_channels = layer.get ('in_channels', default_channels)
            out_channels = layer.get ('out_channels', None)
            if out_channels is None :
                raise Exception ("wrongly defined model")

            filter_size = layer.get ('filter_size', 3)
            stride = layer.get ('stride', 1)
            norm = layer.get ('norm', 'batch_norm')
            activation = layer.get ('activation', 'relu')
            dropout = layer.get ('dropout', None)
            pool_layer = layer.get ('pool_layer', None)
            pool_size = layer.get ('pool_size', 2)
            pool_stride = layer.get ('pool_stride', 1)

            self.layers.append (
                nn.Conv2d (in_channels=in_channels, out_channels=out_channels, kernel_size=filter_size, stride=stride, padding=math.ceil ((filter_size - stride) / 2))
            )
            if norm is not None :
                self.layers.append (
                    self.apply_norm (norm, out_channels)
                )

            if activation is not None :
                self.layers.append (
                    self.apply_activation (activation)
                )

            if pool_layer == 'max_pool' :
                self.layers.append (
                    nn.MaxPool2d (pool_size, pool_stride)
                )
            elif pool_layer == 'avg_pool' :
                self.layers.append (
                    nn.AvgPool2d (pool_size, pool_stride)
                )
            elif pool_layer == 'adaptive_max_pool' :
                self.layers.append (
                    nn.AdaptiveMaxPool2d (pool_size, pool_stride)
                )
            elif pool_layer == 'adaptive_avg_pool' :
                self.layers.append (
                    nn.AdaptiveMaxPool2d (pool_size, pool_stride)
                )

            if dropout is not None :
                self.layers.append (
                    self.apply_dropout (dropout, 2)
                )

            default_channels = out_channels

        self.layers.append (nn.Flatten ())

        fc_layers = model_arguments.get ('fc_layers')
        for layer in fc_layers :
            in_channels = layer.get ('in_channels', None)
            out_channels = layer.get ('out_channels', None)
            norm = layer.get ('norm', None)
            dropout = layer.get ('dropout', None)
            activation = layer.get ('activation', None)

            self.layers.append (
                nn.Linear (in_channels, out_channels)
            )
            if norm is not None :
                self.layers.append (
                    self.apply_norm (norm, out_channels)
                )
            if activation is not None :
                self.layers.append (
                    self.apply_activation (activation)
                )
            if dropout is not None :
                self.layers.append (
                    self.apply_dropout (dropout, 1)
                )

        self.net = nn.Sequential (*self.layers)

    def apply_dropout (self, dropout, layer_count) :
        if layer_count == 1 :
            return nn.Dropout (dropout)
        elif layer_count == 2 :
            return nn.Dropout2d (dropout)
        elif layer_count == 3 :
            return nn.Dropout3d (dropout)
        else :
            print ("Not supported Dropout")

    def apply_activation (self, activation) :
        if activation == 'relu' :
            return nn.ReLU ()
        elif activation == 'sigmoid' :
            return nn.Sigmoid ()
        elif activation == 'leaky_relu' :
            return nn.LeakyReLU ()
        elif activation == 'prelu' :
            return nn.PReLU ()
        elif activation == 'tanh' :
            return nn.Tanh ()
        elif activation == 'elu' :
            return nn.ELU ()
        elif activation == 'harshrink' :
            return nn.Hardshrink ()
        elif activation == 'hard_sigmoid' :
            return nn.Hardsigmoid ()
        elif activation == 'hard_tanh' :
            return nn.Hardtanh ()
        elif activation == 'hard_swish' :
            return nn.Hardswish ()
        elif activation == 'log_sigmoid' :
            return nn.LogSigmoid ()
        elif activation == 'multi_head_attention' :
            return nn.MultiheadAttention ()
        elif activation == 'relu6' :
            return nn.ReLU6 ()
        elif activation == 'selu' :
            return nn.SELU ()
        elif activation == 'celu' :
            return nn.CELU ()
        elif activation == 'gelu' :
            return nn.GELU ()
        elif activation == 'silu' :
            return nn.SiLU ()
        elif activation == 'mish' :
            return nn.Mish ()
        elif activation == 'softplus' :
            return nn.Softplus ()
        elif activation == 'softshrink' :
            return nn.Softshrink ()
        elif activation == 'softsign' :
            return nn.Softsign ()
        elif activation == 'tanshrink' :
            return nn.Tanhshrink ()
        elif activation == 'threshold' :
            return nn.Threshold ()
        elif activation == 'softmax2d' :
            return nn.Softmax2d ()
        elif activation == 'log_softmax' :
            return nn.LogSoftmax ()
        elif activation == 'softmax' :
            return nn.Softmax ()
        elif activation == 'softmin' :
            return nn.Softmin ()
        elif activation == 'adaptive_softmax' :
            return nn.AdaptiveLogSoftmaxWithLoss ()
        else :
            print ("Activation not supported!!")

    def apply_norm (self, norm, out_channels) :
        if norm == 'batch_norm' :
            return nn.BatchNorm2d (out_channels)
        elif norm == 'batch_norm1d' :
            return nn.BatchNorm1d (out_channels)
        elif norm == 'batch_norm3d' :
            return nn.BatchNorm3d (out_channels)
        elif norm == 'lazy_batch_norm' :
            return nn.LazyBatchNorm2d ()
        elif norm == 'lazy_batch_norm1d' :
            return nn.LazyBatchNorm1d ()
        elif norm == 'lazy_batch_norm3d' :
            return nn.LazyBatchNorm3d ()
        elif norm == 'sync_batch_norm' :
            return nn.SyncBatchNorm (out_channels)
        elif norm == 'instance_norm' :
            return nn.InstanceNorm2d (out_channels)
        elif norm == 'instance_norm1d' :
            return nn.InstanceNorm1d (out_channels)
        elif norm == 'instance_norm3d' :
            return nn.InstanceNorm3d (out_channels)
        else :
            print ("Not supported Normalization")
        
    def forward (self, x) :
        x = self.net (x)

        return x

class CNN :
    def __init__ (self, dataset_arguments, model_arguments) :
        self.dataset_arguments = dataset_arguments
        self.model_arguments = model_arguments

        self.train_file_path = self.dataset_arguments.get ("train_file", None)
        self.test_file_path = self.dataset_arguments.get ("test_file", None)
        self.batch_size = self.dataset_arguments.get ("batch_size", 1)

        if self.train_file_path is None :
            raise Exception ("need the path to the training file")

        self.train_data_loader = self._create_dataloader (self.train_file_path, is_train=True)
        self.test_data_loader = self._create_dataloader (self.test_file_path, is_train=False)

        self.model = Model (self.model_arguments)
        self.optimizer = self.model_arguments.get ('optimizer', 'adam')
        self.learning_rate = self.model_arguments.get ('learning_rate', 1e-4)
        if self.optimizer == 'adam' :
            self.optimizer = torch.optim.Adam (list (self.model.parameters ()), lr=self.learning_rate)
        else :
            raise Exception ("Please define a valid optimizer, currently valid inputs -> (adam)")

        self.criteria = self.model_arguments.get ('loss_criteria', 'cross_entropy')
        if self.criteria == 'cross_entropy' :
            self.criteria = nn.CrossEntropyLoss ()
        elif self.criteria == 'binary_cross_entropy_with_logits' :
            self.criteria = nn.BCEwithLogits ()
        elif self.criteria == 'binary_cross_entropy' :
            self.criteria = nn.BCE ()

        print (self.model)


    def _create_dataloader (self, filename, is_train=True) :
        if filename is None :
            return None

        dataloader = Dataset (self.dataset_arguments, filename)
        dataloader = data.DataLoader (dataset=dataloader, batch_size=self.batch_size, shuffle=True)
        
        return dataloader

    def fit (self, epochs, result_dir, des_file, evaluate_test=False) :
        for epoch in range (epochs) :
            print (f'Running Epoch: {epoch}')

            avg_loss = 0.0
            self.model.train ()
            for i, data in enumerate (self.train_data_loader) :
                image, label = data
                self.optimizer.zero_grad ()

                output = self.model (image)

                loss = self.criteria (output, label)
                loss.backward ()
                self.optimizer.step ()

                avg_loss += loss.item ()

            print (f"Average Loss {avg_loss / (i+1)}")

            if evaluate_test == True :
                self.model.eval ()
                accuracy = 0
                for i, data in enumerate (self.test_data_loader) :
                    image, label = data
                    
                    output = self.model (image)
                    output = np.argmax (output.data.numpy (), axis=1)

                    label = label.data.numpy ()

                    accuracy += np.sum (output == label)

                print (f"Accuracy on Test Set: {accuracy / ((i+1)*self.batch_size)}")

        if not os.path.isdir (result_dir) :
            os.mkdir (result_dir)
        torch.save (self.model.state_dict (), os.path.join (result_dir, 'model.pkl'))

        self.model.eval ()
        accuracy = 0
        for i, data in enumerate (self.test_data_loader) :
            image, label = data
            
            output = self.model (image)
            output = np.argmax (output.data.numpy (), axis=1)

            label = label.data.numpy ()

            accuracy += np.sum (output == label)

        accuracy = accuracy / ((i+1)*self.batch_size)

        fit_description = {
            "model": str (self.model),
            "arguments": self.model_arguments if self.model_arguments else "default",
            "type": self.model_arguments["type"],
            "algorithm": self.model_arguments["algorithm"],
            "dataset_props": self.dataset_arguments,
            "model_props": self.model_arguments,
            "results_path": str(result_dir),
            "model_path": str(os.path.join (result_dir, "model.pkl")),
            "results_on_test_data": str (accuracy),
        }

        with open(des_file, "w", encoding="utf-8") as f:
            json.dump(fit_description, f, ensure_ascii=False, indent=4)

    def evaluate (self, model_path=None) :
        if model_path is not None :
            if not os.path.isdir (model_path) :
                raise Exception ("Not a valid directory")
            self.model.load (os.path.join (model_path, 'model.pkl'))

        self.model.eval ()
        accuracy = 0
        for i, data in enumerate (self.test_data_loader) :
            image, label = data
            
            output = self.model (image)
            output = np.argmax (output.data.numpy (), axis=1)

            if output == label :
                accuracy += 1

        print (f"Accuracy on Test Set: {accuracy / ((i+1)*self.batch_size)}")

# Just for testing these functions
if __name__ == '__main__' :
    yaml_file = open ('../tests/test_igel/igel_files/igel_cnn.yaml')
    yaml_file = yaml.safe_load (yaml_file)

    if not os.path.isdir ('testing_cnn') :
        os.mkdir ("testing_cnn")

    cnn = CNN (yaml_file['dataset'], yaml_file['model'])
    cnn.fit (10, "testing", True)
    cnn.evaluate ("testing")