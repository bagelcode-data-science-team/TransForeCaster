import os 
import argparse
from src.model import TransForeCaster
from src.utils import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TransForeCaster Training')
    parser.add_argument('--data', type=str, default='./src/data/', help='data directory')
    parser.add_argument('--lr', type=int, default=0.001, help='learning rate')
    parser.add_argument('--batch', type=int, default=64, help='batch size')
    parser.add_argument('--input', type=int, default=7, help='input day')
    parser.add_argument('--target', type=int, default=14, help='target day')
    parser.add_argument('--epoch', type=int, default=15, help='epoch')
    parser.add_argument('--objective', type=str, default='purchase', help='objective (purchase or churn)')
    args = parser.parse_args()

    # Load data
    train_data, valid_data, test_data = load_data(args.data)
    train_gen = DataGenerator(train_data, args.input, args.target, args.batch)
    valid_gen = DataGenerator(train_data, args.input, args.target, args.batch) 
    test_gen = DataGenerator(test_data, args.input, args.target, args.batch)

    # Pretrain the encoder & Train the model 
    encoding_layers = pretrain_encoder(train_gen)
    model = TransForeCaster(behavior_length=train_gen.X[1].shape[-1],
                portrait_length=train_gen.X[2].shape[-1],
                input_days=args.input,
                target_day=args.target,
                vocab_size_dict={key: len(encoder.label_encoder.classes_) for key, encoder in test_gen.encoders.items()},
                behavior_category_indice=test_gen.behavior_indices,
                portrait_category_indice=test_gen.portrait_indices,
                encoding_layers=encoding_layers,
                )
    train(model, args.lr, args.epoch, args.objective, train_gen, valid_gen) 
    if args.objective == 'purchase':
        evaluate_purchase(model, test_gen)
    else:
        evaluate_churn(model, test_gen)

    


    

    