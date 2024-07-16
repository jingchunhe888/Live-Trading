from new import *
from lumibot.strategies import Strategy
from lumibot.brokers import Alpaca
from lumibot.traders import Trader
from config import ALPACA_CONFIG

# MAIN DRIVER CODE
class arima(Strategy):
    def initialize(self):
        self.sleeptime = '1M'
        self.symbols = ["GOOG"]

    def on_trading_iteration(self):
        symbol ="GOOG"
        pos = self.get_position(symbol)
        predict = run()

        price = self.get_last_price(symbol)
        action = predict > price
        quantity = self.cash // price

        #previous = None
        #wanted to minimize trade so least transactions code.
        #if previous trade movement is same as current, don't need to
        #sell and buy again
        if pos is None:
            if action:
                order = self.create_order(symbol, quantity, "buy")
                self.submit_order(order)
                #previous = True

            if not action:
                pass
                #no short option available
                #quantity = self.cash // price
                #order = self.create_order(symbol, quantity, "short")
                #self.submit_order(order)
                #previous = False

        else:
            self.sell_all()
            #doing this so that there are more transactions on alpaca
            #and making sure that new predictions are made every minute
            if action:
                order = self.create_order(symbol, quantity, "buy")
                self.submit_order(order)
            else:
                pass
       #     if action == previous:
      #      self.sell_all()




if __name__ == "__main__":
    broker = Alpaca(ALPACA_CONFIG)
    strategy = arima(broker=broker)
    trader = Trader()
    trader.add_strategy(strategy)
    trader.run_all()

'''
                    if action:
                        quantity = self.cash // price
                        order = self.create_order(symbol, quantity, "buy")
                        self.submit_order(order)
                        previous = True
                    else:
                        pass
                        #quantity = self.cash // price
                        #order = self.create_order(symbol, quantity, "short")
                        #self.submit_order(order)
                        #previous = False
                        
                        '''