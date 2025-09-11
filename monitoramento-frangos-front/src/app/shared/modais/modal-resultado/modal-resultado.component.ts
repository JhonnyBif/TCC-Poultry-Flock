import { CommonModule } from '@angular/common';
import { Component, Inject } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { MatButtonModule } from '@angular/material/button';
import { MatCardModule } from '@angular/material/card';
import { MatCheckboxModule } from '@angular/material/checkbox';
import {
  MAT_DIALOG_DATA,
  MatDialog,
  MatDialogModule,
} from '@angular/material/dialog';
import { MatFormFieldModule } from '@angular/material/form-field';
import { MatIconModule } from '@angular/material/icon';
import { MatInputModule } from '@angular/material/input';
import { MatListModule } from '@angular/material/list';
import { MatRadioModule } from '@angular/material/radio';
import { MatTableModule } from '@angular/material/table';
import { NgxFileDropModule } from 'ngx-file-drop';
import { ImageModalComponent } from '../image-modal/image-modal.component';
import { BreakpointObserver } from '@angular/cdk/layout';

@Component({
  selector: 'app-modal-resultado',
  standalone: true,
  imports: [
    FormsModule,
    CommonModule,
    MatRadioModule,
    MatIconModule,
    MatButtonModule,
    MatFormFieldModule,
    MatCheckboxModule,
    MatInputModule,
    NgxFileDropModule,
    MatCardModule,
    MatListModule,
    MatTableModule,
    MatDialogModule,
  ],
  templateUrl: './modal-resultado.component.html',
  styleUrls: ['./modal-resultado.component.scss'],
})
export class ModalResultadoComponent {
  imagemExpandida: string | null = null;
  mostrarQuantidadePorMetroQuadrado: boolean = false;
  constructor(
    @Inject(MAT_DIALOG_DATA)
    public data: {
      quantidade_por_metro_quadrado:any
    },
    private dialog: MatDialog,
    private breakpointObserver: BreakpointObserver
  ) {}
  abrirImagem(img: string) {
    const isMobile = this.breakpointObserver.isMatched('(max-width: 600px)');
    this.dialog.open(ImageModalComponent, {
      data: { imagem: img },
      width: isMobile ? '100vw' : '90vw',
      minWidth: isMobile ? '100vw' : '90vw',
      maxWidth: isMobile ? '100vw' : '90vw',
      height: '90vh',
      panelClass: isMobile ? 'full-screen-modal' : 'custom-modal',
    });
  }
  downloadImage(data: string) {
    const byteCharacters = atob(data);
    const byteNumbers = new Array(byteCharacters.length);
    for (let i = 0; i < byteCharacters.length; i++) {
      byteNumbers[i] = byteCharacters.charCodeAt(i);
    }
    const byteArray = new Uint8Array(byteNumbers);
    let blob = new Blob([byteArray], { type: 'image/png' });
    const link = document.createElement('a');
    link.href = window.URL.createObjectURL(blob);
    link.download = 'Image';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  }
}
