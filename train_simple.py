
env = TSPEnv(generator_params={'num_loc': 20})

policy = DeepACO(env_name=env.name,
                           temperature=1,
                           top_p=1,
                           n_ants=20,
                           top_k=50,
                           embed_dim=128,
                           num_layers_graph_encoder=3)

model = REINFORCE(env=env,
                    policy=policy,
                    baseline="rollout",
                    batch_size=512,
                    train_data_size=100_000,
                    val_data_size=10_000,
                    dataloader_num_workers=0,
                    optimizer_kwargs={"lr": 1e-4})

trainer = RL4COTrainer(
    accelerator="mps",
    max_epochs=1,
    devices=1,
    logger=None
)

trainer.fit(model) # <----ERROR OCCURRED HERE
