import { Module } from '@nestjs/common';
import { AppController } from './app.controller';
import { AppService } from './app.service';
import { LogRatingModule } from './log-rating/log-rating.module';
import { OutputModule } from './output/output.module';

@Module({
  imports: [LogRatingModule, OutputModule],
  controllers: [AppController],
  providers: [AppService],
})
export class AppModule {}
