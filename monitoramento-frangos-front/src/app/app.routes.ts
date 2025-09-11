import { Routes } from '@angular/router';
import { VideoProcessorComponent } from './pages/video-processor/video-processor.component';

export const routes: Routes = [
  { path: '', redirectTo: '', pathMatch: 'full' },
  { path: '', component: VideoProcessorComponent },
  { path: '**', redirectTo: '' }
];
