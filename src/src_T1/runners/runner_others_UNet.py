from src_T1.runners.runner_base import Runner_Base


class Runner_UNet(Runner_Base):

    def forward(self, data, model):
        X, y = data
        p, _, emb = model(X)

        return {'p':p, 'emb':emb, 'p_rate':p, 'y':y}

    def forward_loss(self, res):
        p, emb, y = res['p'], res['emb'], res['y']
        loss_dict = {}
        loss_dict['GUL'] = self.GUL_func(p, y[:, [0]], None)
        loss_dict['BCE'] = self.BCE_func(p, y[:, [0]], None)
        loss_final = loss_dict['GUL']*self.hp.loss_weight['GUL'] + loss_dict['BCE']*self.hp.loss_weight['BCE']

        return loss_dict, loss_final
