from transformers.integrations import NeptuneCallback
from transformers import GPT2LMHeadModel, TrainingArguments, Trainer
from transformers import DataCollatorForLanguageModeling
from data_preprocessing import GPT2Dataset


def suppress_neptune(trainer):
    for cb in trainer.callback_handler.callbacks:
        if isinstance(cb, NeptuneCallback):
            trainer.callback_handler.remove_callback(cb)



def train_gpt2(train_tokenized, val_tokenized, tokenizer, args):
    num_epochs = args.num_epochs
    train_dataset = GPT2Dataset(train_tokenized)
    val_dataset = GPT2Dataset(val_tokenized)
    
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    model.resize_token_embeddings(len(tokenizer))
    batch_size = args.batch_size
    
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False,
    )

    training_args = TrainingArguments(
        output_dir='./results',          # output directory
        overwrite_output_dir=True,
        num_train_epochs=num_epochs,              # total number of training epochs
        per_device_train_batch_size=batch_size,   # batch size per device during training
        per_device_eval_batch_size=batch_size,    # batch size for evaluation
        warmup_steps=500,                # number of warmup steps for learning rate scheduler
        weight_decay=0.01,               # strength of weight decay
        logging_dir='./logs',            # directory for storing logs
        logging_steps=10,
        learning_rate=5e-4,
        save_steps=1000,
        fp16=True,
        save_total_limit=1,
        lr_scheduler_type="cosine",
        evaluation_strategy='steps',
        eval_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model='eval_loss',
    )
    
    trainer = Trainer(
        model=model,                         # the instantiated 🤗 Transformers model to be trained
        args=training_args,                  # training arguments, defined above
        data_collator=data_collator,
        train_dataset=train_dataset,         # training dataset
        eval_dataset=val_dataset,            # evaluation dataset
        tokenizer=tokenizer,
    )
    suppress_neptune(trainer)
    
    trainer.train()
    trainer.save_model('gpt2')
    trainer.evaluate()
    trainer.save_model('gpt2')
    print('Done!')