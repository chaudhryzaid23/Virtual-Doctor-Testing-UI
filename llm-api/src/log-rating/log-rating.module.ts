import { Module } from '@nestjs/common';
import { LogRatingService } from './log-rating.service';
import { LogRatingController } from './log-rating.controller';

@Module({
  controllers: [LogRatingController],
  providers: [LogRatingService],
})
export class LogRatingModule {}
